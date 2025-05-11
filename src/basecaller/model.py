import types
from bonito.crf.model import CTC_CRF, SeqdistModel
from bonito.nn import (
    Convolution,
    LinearCRFEncoder,
    LinearUpsample,
    Permute,
    Serial,
    NamedSerial,
    Stack,
)
from bonito.transformer.model import TransformerEncoderLayer, use_koi


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
            TransformerEncoderLayer(
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
            "transormer_encoder": build_transformer_encoder(),
            "upsample": LinearUpsample(512, scale_factor=2),
            "crf": build_crf_encoder(),
        }
    )
    seqdist = CTC_CRF(state_len=5, alphabet=["A", "C", "G", "T", "N"])
    model = SeqdistModel(encoder=encoder, seqdist=seqdist)
    model.use_koi = types.MethodType(use_koi, model)
    return model
