from bonito.nn import Convolution, Linear, Permute, Serial
from torch.nn import Module


class HomopolymerFeatureExtractor(Module):
    def __init__(
        self,
        insize,
        hp_feature_dim,
        num_bases=5,
        activation="swish",
        norm="batchnorm",
    ):
        super().__init__()
        self.insize = insize
        self.hp_feature_dim = hp_feature_dim
        self.num_bases = num_bases

        # Convolutional layers extract features but maintain length
        self.extractor = Serial(
            [
                Convolution(
                    insize=insize,
                    size=64,
                    winlen=3,
                    padding=1,
                    activation=activation,
                    norm=norm,
                ),
                Convolution(
                    insize=64,
                    size=128,
                    winlen=3,
                    padding=1,
                    activation=activation,
                    norm=norm,
                ),
                Convolution(
                    insize=128,
                    size=hp_feature_dim,
                    winlen=1,
                    padding=0,
                    activation=activation,
                    norm=norm,
                ),
            ]
        )

        # Conv output is (N, C, L) but linear layers take in (N, L, C)
        self.permute = Permute([0, 2, 1])

        # Prediction heads
        self.hp_length_predictor = Linear(in_features=hp_feature_dim, out_features=1)
        self.is_hp_predictor = Linear(in_features=hp_feature_dim, out_features=1)
        self.hp_base_predictor = Linear(
            in_features=hp_feature_dim, out_features=num_bases
        )

    def forward(self, x):
        hp_features = self.extractor(x)
        hp_features_permuted = self.permute(hp_features)
        hp_length_logits = self.hp_length_predictor(hp_features_permuted)
        is_hp_logits = self.is_hp_predictor(hp_features_permuted)
        hp_bases_logits = self.hp_base_predictor(hp_features_permuted)

        return hp_features, hp_length_logits, is_hp_logits, hp_bases_logits
