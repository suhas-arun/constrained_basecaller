import types
import logging

from basecaller.hp_extractor import HomopolymerFeatureExtractor

logger = logging.getLogger(__name__)

import torch
import torch.nn.functional as F
from torch.nn import Module, AvgPool1d
from bonito.crf.model import CTC_CRF, SeqdistModel
from bonito.nn import (
    Convolution,
    Linear,
    LinearCRFEncoder,
    LinearUpsample,
    Permute,
    Serial,
    Stack,
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


class HomopolymerAwareEncoder(Module):
    def __init__(self, hp_feature_dim=128, num_bases=5, transformer_d_model=512):
        super().__init__()
        self.hp_feature_dim = hp_feature_dim
        self.num_bases = num_bases

        # Convolutional layers
        self.initial_convs = self.build_initial_convs()  # output length L/3
        self.final_convs = self.build_final_convs()  # output length L/12
        self.main_permute = Permute([0, 2, 1])  # (N, C, L) -> (N, L, C)

        # Homopolymer feature extraction
        self.hp_extractor = HomopolymerFeatureExtractor(128, hp_feature_dim, num_bases)
        self.hp_downsample = AvgPool1d(kernel_size=4, stride=4)  # L/3 -> L/12
        self.hp_permute = Permute([0, 2, 1])  # (N, C, L) -> (N, L, C)
        self.hp_project = Linear(hp_feature_dim, transformer_d_model)

        self.transformer_encoder = self.build_transformer_encoder()
        self.upsample = LinearUpsample(transformer_d_model, scale_factor=2)
        self.crf_encoder = self.build_crf_encoder()

    def forward(self, x, hp_true_labels=None):
        x = x.unsqueeze(1)
        # Initial convolutions
        x_intermediate = self.initial_convs(x)  # (N, 128, L/3)

        # Homopolymer feature extraction
        hp_features, hp_lengths_logits, is_hp_logits, hp_bases_logits = (
            self.hp_extractor(x_intermediate)
        )
        hp_features = self.hp_downsample(hp_features)  # (N, 128, L/12)
        hp_features = self.hp_permute(hp_features)  # (N, L/12, 128)
        hp_features = self.hp_project(hp_features)  # (N, L/12, 512)

        # Main convolutional path
        main_features = self.final_convs(x_intermediate)  # (N, 512, L/12)
        main_features = self.main_permute(main_features)  # (N, L/12, 512)

        # Resize homopolymer features to match main features length
        main_features_length = main_features.shape[1]
        hp_features_length = hp_features.shape[1]

        if main_features_length != hp_features_length:
            # (N, L, C) -> (N, C, L) for interpolation
            hp_features_to_resize = hp_features.permute(0, 2, 1)
            hp_features_resized = F.interpolate(
                hp_features_to_resize,
                size=main_features_length,
                mode="linear",
                align_corners=False,
            )
            # (N, C, L) -> (N, L, C) for fusion
            hp_features = hp_features_resized.permute(0, 2, 1)

        # Fuse homopolymer features with main features
        fused_features = main_features + hp_features

        transformer_output = self.transformer_encoder(fused_features)
        upsampled_output = self.upsample(transformer_output)
        logits = self.crf_encoder(upsampled_output)

        outputs = {
            "logits": logits,
            "hp_lengths_logits": hp_lengths_logits,
            "is_hp_logits": is_hp_logits,
            "hp_bases_logits": hp_bases_logits,
        }

        if hp_true_labels is not None:
            outputs["hp_true_labels"] = hp_true_labels

        return outputs

    def build_initial_convs(self):
        return Serial(
            [
                Convolution(1, 64, 5, padding=2, activation="swish", norm="batchnorm"),
                Convolution(64, 64, 5, padding=2, activation="swish", norm="batchnorm"),
                Convolution(
                    64,
                    128,
                    9,
                    stride=3,
                    padding=4,
                    activation="swish",
                    norm="batchnorm",
                ),
            ]
        )

    def build_final_convs(self):
        return Serial(
            [
                Convolution(
                    128,
                    128,
                    9,
                    stride=2,
                    padding=4,
                    activation="swish",
                    norm="batchnorm",
                ),
                Convolution(
                    128,
                    512,
                    5,
                    stride=2,
                    padding=2,
                    activation="swish",
                    norm="batchnorm",
                ),
            ]
        )

    def build_transformer_encoder(self):
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

    def build_crf_encoder(self):
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


def loss_fn(self, outputs, ctc_targets, ctc_target_lengths):
    hp_true_labels = outputs.get("hp_true_labels", None)

    # Extract logits and auxiliary outputs
    logits = outputs["logits"]
    hp_lengths_logits = outputs["hp_lengths_logits"]
    is_hp_logits = outputs["is_hp_logits"]
    hp_bases_logits = outputs["hp_bases_logits"]

    device = logits.device

    aux_loss_weight = 0.1

    # Initialise losses
    ctc_loss = torch.tensor(0.0, device=device)
    hp_lengths_loss = torch.tensor(0.0, device=device)
    is_hp_loss = torch.tensor(0.0, device=device)
    hp_bases_loss = torch.tensor(0.0, device=device)

    if hp_true_labels is not None:
        true_hp_lengths = hp_true_labels["hp_lengths"].to(device)
        true_is_hp = hp_true_labels["is_hp"].to(device)
        true_hp_bases = hp_true_labels["hp_bases"].to(device)

        # Regression loss for homopolymer lengths
        hp_lengths_logits = hp_lengths_logits.squeeze(-1) 
        hp_lengths_loss = F.mse_loss(hp_lengths_logits, true_hp_lengths)

        # Binary cross-entropy loss for homopolymer presence
        is_hp_logits = is_hp_logits.squeeze(-1)
        is_hp_loss = F.binary_cross_entropy_with_logits(
            is_hp_logits, true_is_hp.float()
        )

        # Categorical cross-entropy loss for homopolymer bases
        true_hp_bases = true_hp_bases.long()
        hp_bases_loss = F.cross_entropy(
            hp_bases_logits.view(-1, self.encoder.num_bases),
            true_hp_bases.view(-1),
            # 'N' is the last base
            ignore_index=self.encoder.num_bases - 1,
        )
        aux_loss = hp_lengths_loss + is_hp_loss + hp_bases_loss
        final_loss = aux_loss_weight * aux_loss
    else:
        ctc_targets = ctc_targets.to(device)
        ctc_target_lengths = ctc_target_lengths.to(device)
        logits = logits.to(torch.float32)
        ctc_loss = self.seqdist.ctc_loss(logits, ctc_targets, ctc_target_lengths)
        final_loss = ctc_loss

    return {
        "loss": final_loss,
        "ctc_loss": ctc_loss,
        "hp_lengths_loss": hp_lengths_loss,
        "is_hp_loss": is_hp_loss,
        "hp_bases_loss": hp_bases_loss,
    }


def ConstraintAwareBasecaller():
    alphabet = ["A", "C", "G", "T", "N"]
    encoder = HomopolymerAwareEncoder(
        hp_feature_dim=128, num_bases=len(alphabet), transformer_d_model=512
    )
    seqdist = CTC_CRF(state_len=5, alphabet=alphabet)
    model = SeqdistModel(encoder=encoder, seqdist=seqdist)
    model.use_koi = types.MethodType(use_koi, model)
    model.loss = types.MethodType(loss_fn, model)
    return model
