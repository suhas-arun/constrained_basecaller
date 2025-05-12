import os
from bonito.data import ComputeSettings, DataSettings, ModelSetup, load_data
from bonito.training import Trainer
import toml
import torch

from basecaller.model import ConstraintAwareBasecaller
from utils import save_model_weights


def main(args):
    workdir = os.path.expanduser(args.training_directory)
    os.makedirs(workdir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("[initializing custom model]")
    model = ConstraintAwareBasecaller()
    model = model.to(device)

    config = {
        "standardisation": {},  # Update this if you're using signal normalization
        "optim": {"betas": [0.9, 0.999], "eps": 1e-08, "weight_decay": 0.0},
    }

    argsdict = dict(training=vars(args))
    argsdict["training"]["pwd"] = os.getcwd()
    toml.dump({**config, **argsdict}, open(os.path.join(workdir, "config.toml"), "w"))

    data = DataSettings(
        training_data=args.directory,
        num_train_chunks=args.chunks,
        num_valid_chunks=args.valid_chunks,
        output_dir=workdir,
    )
    model_setup = ModelSetup(
        n_pre_context_bases=getattr(model, "n_pre_context_bases", 0),
        n_post_context_bases=getattr(model, "n_post_context_bases", 0),
        standardisation=config["standardisation"],
    )
    compute_settings = ComputeSettings(
        batch_size=args.batch,
        num_workers=4,
        seed=25,
    )

    print("[loading data]")
    train_loader, valid_loader = load_data(data, model_setup, compute_settings)

    trainer = Trainer(
        model=model,
        device=device,
        train_loader=train_loader,
        valid_loader=valid_loader,
        quantile_grad_clip=True,
        chunks_per_epoch=args.chunks,
        batch_size=args.batch,
    )

    lr = (
        float(args.lr) if "," not in args.lr else [float(x) for x in args.lr.split(",")]
    )
    trainer.fit(workdir, args.epochs, lr, **config["optim"])

    save_model_weights(model, args.weights_path)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training-directory",
        type=str,
        help="Directory to save training data and checkpoints",
    )
    parser.add_argument(
        "--directory",
        type=str,
        help="Directory containing training data",
    )
    parser.add_argument(
        "--chunks",
        type=int,
        default=1000,
        help="Number of chunks to use for training",
    )
    parser.add_argument(
        "--valid-chunks",
        type=int,
        default=100,
        help="Number of chunks to use for validation",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=32,
        help="Batch size for training",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs to train for",
    )
    parser.add_argument(
        "--lr",
        type=str,
        default="0.001",
        help="Learning rate for training",
    )
    parser.add_argument(
        "--weights-path",
        type=str,
        help="Path to save model weights",
    )
    args = parser.parse_args()

    main(args)
