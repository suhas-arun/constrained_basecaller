import types
import logging

logger = logging.getLogger(__name__)

import torch
import torch.nn.functional as F
from bonito.crf.model import CTC_CRF, SeqdistModel
from bonito.nn import (
    Convolution,
    LinearCRFEncoder,
    LinearUpsample,
    Permute,
    Serial,
    NamedSerial,
    Stack,
    Module,
)
from bonito.transformer.model import (
    TransformerEncoderLayer,
    sliding_window_mask,
    use_koi,
)

try:
    from flash_attn import flash_attn_qkvpacked_func
    from flash_attn.layers.rotary import RotaryEmbedding
except ImportError:
    logger.warning(
        "please install flash-attn to use the transformer module: "
        "`pip install flash-attn --no-build-isolation`"
    )


class ConstraintAwareAttention(Module):
    def __init__(
        self,
        d_model,
        nhead,
        qkv_bias=False,
        out_bias=True,
        rotary_dim=None,
        attn_window=None,
    ):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.rotary_dim = self.head_dim if rotary_dim is None else rotary_dim

        self.Wqkv = torch.nn.Linear(d_model, 3 * d_model, bias=qkv_bias)
        self.out_proj = torch.nn.Linear(d_model, d_model, bias=out_bias)

        self.rotary_emb = RotaryEmbedding(self.rotary_dim, interleaved=False)
        self.attn_window = (-1, -1) if attn_window is None else tuple(attn_window)

    def attn_func(self, qkv):
        if torch.cuda.get_device_capability(qkv.device)[0] >= 8 and (
            torch.is_autocast_enabled() or qkv.dtype == torch.half
        ):
            attn_output = flash_attn_qkvpacked_func(qkv, window_size=self.attn_window)
        else:
            q, k, v = torch.chunk(qkv.permute(0, 2, 3, 1, 4), chunks=3, dim=1)
            # q, k, v all have shape [64, 1, 8, 333, 64]
            mask = sliding_window_mask(qkv.shape[1], self.attn_window, q.device)
            attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            attn_output = attn_output.permute(0, 1, 3, 2, 4)

        return attn_output

    def forward(self, x):
        N, T, _ = x.shape

        qkv = self.Wqkv(x).view(N, T, 3, self.nhead, self.head_dim)

        qkv = self.rotary_emb(qkv)

        attn_output = self.attn_func(qkv).reshape(N, T, self.d_model)

        out = self.out_proj(attn_output)

        return out


class ConstraintAwareTransformerLayer(TransformerEncoderLayer):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward,
        deepnorm_alpha,
        deepnorm_beta,
        attn_window=None,
    ):
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            deepnorm_alpha=deepnorm_alpha,
            deepnorm_beta=deepnorm_beta,
            attn_window=attn_window,
        )
        self.self_attn = ConstraintAwareAttention(
            d_model=d_model,
            nhead=nhead,
            qkv_bias=False,
            out_bias=True,
            attn_window=attn_window,
        )


def build_conv_layers():
    return Serial(
        sublayers=[
            Convolution(1, 64, 5, padding=2, activation="swish", norm="batchnorm"),
            Convolution(64, 64, 5, padding=2, activation="swish", norm="batchnorm"),
            Convolution(
                64, 128, 9, stride=3, padding=4, activation="swish", norm="batchnorm"
            ),
            Convolution(
                128, 128, 9, stride=2, padding=4, activation="swish", norm="batchnorm"
            ),
            Convolution(
                128, 512, 5, stride=2, padding=2, activation="swish", norm="batchnorm"
            ),
            Permute([0, 2, 1]),
        ]
    )


def build_transformer_encoder():
    return Stack(
        sublayers=[
            ConstraintAwareTransformerLayer(
                d_model=512,
                nhead=8,
                dim_feedforward=2048,
                deepnorm_alpha=2.4494897,
                deepnorm_beta=0.2886751,
                attn_window=[
                    127,
                    128,
                ],
            )
            for _ in range(18)
        ]
    )


def build_crf_encoder():
    return LinearCRFEncoder(
        insize=512,
        n_base=4,
        state_len=5,
        bias=False,
        scale=5.0,
        blank_score=2.0,
        expand_blanks=True,
        permute=[
            1,
            0,
            2,
        ],
    )


def ConstraintAwareBasecaller():
    encoder = NamedSerial(
        {
            "conv": build_conv_layers(),
            "transformer_encoder": build_transformer_encoder(),
            "upsample": LinearUpsample(512, scale_factor=2),
            "crf": build_crf_encoder(),
        }
    )
    seqdist = CTC_CRF(state_len=5, alphabet=["A", "C", "G", "T", "N"])
    model = SeqdistModel(encoder=encoder, seqdist=seqdist)
    model.use_koi = types.MethodType(use_koi, model)
    model.forward = types.MethodType(forward, model)
    return model
