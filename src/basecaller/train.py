from collections import OrderedDict
import os
from bonito.data import ComputeSettings, DataSettings, ModelSetup, load_data
from bonito.training import Trainer
import numpy as np
import toml
import torch
from torch.utils.data import Subset, DataLoader

from basecaller.model import ConstraintAwareBasecaller
from basecaller.pretrain_dataset import PretrainDataset, hp_collate_fn
from utils import save_model_weights


def main(args):
    workdir = os.path.expanduser(args.output_directory)
    os.makedirs(workdir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("[Initialising custom model]")
    model = ConstraintAwareBasecaller()
    model = model.to(device)

    config = {
        "standardisation": {},  # Update this if you're using signal normalization
        "optim": {"betas": [0.9, 0.999], "eps": 1e-08, "weight_decay": 0.0},
    }

    argsdict = dict(training=vars(args))
    toml.dump({**config, **argsdict}, open(os.path.join(workdir, "config.toml"), "w"))

    if args.pre_training:
        print("[Stage 1: Pre-training homopolymer feature extractor]")
        print("[Freezing layers in the basecaller model]")
        for name, param in model.named_parameters():
            if "hp_extractor" not in name and "initial_convs" not in name:
                param.requires_grad = False

        print("[Loading pre-training dataset]")
        pretrain_dataset = PretrainDataset(
            fasta_file=args.pre_input_fasta,
            fast5_dir=args.pre_input_fast5_dir,
            chunksize=args.pre_chunksize,
            overlap=args.pre_overlap,
            max_homopolymer_length=args.pre_max_homopolymer_length,
        )

        num_chunks = len(pretrain_dataset)
        num_valid_chunks = int(num_chunks * 0.1)
        num_train_chunks = num_chunks - num_valid_chunks

        indices = np.arange(num_chunks)
        np.random.shuffle(indices)

        train_indices = indices[:num_train_chunks]
        valid_indices = indices[num_train_chunks : num_train_chunks + num_valid_chunks]

        train_dataset = Subset(pretrain_dataset, train_indices)
        valid_dataset = Subset(pretrain_dataset, valid_indices)

        train_loader = DataLoader(
            train_dataset, batch_size=args.batch, shuffle=True, collate_fn=hp_collate_fn
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=args.batch,
            shuffle=False,
            collate_fn=hp_collate_fn,
        )

        chunks_per_epoch = len(train_loader)
    else:
        print("[Stage 2: Training basecaller model]")
        if args.pre_weights_path:
            print(f"[Loading pre-trained weights from {args.pre_weights_path}]")
            weights = torch.load(args.pre_weights_path, map_location=device)
            initial_convs_weights = get_layer_weights(weights, "encoder.initial_convs.")
            hp_extractor_weights = get_layer_weights(weights, "encoder.hp_extractor.")

            model.encoder.initial_convs.load_state_dict(initial_convs_weights)
            model.encoder.hp_extractor.load_state_dict(hp_extractor_weights)

        # TODO: finish stage 2 training setup

        data = DataSettings(
            training_data=args.training_directory,
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
        chunks_per_epoch=chunks_per_epoch,
        batch_size=args.batch,
        pre_training=args.pre_training,
    )

    lr = (
        float(args.lr) if "," not in args.lr else [float(x) for x in args.lr.split(",")]
    )
    trainer.fit(workdir, args.epochs, lr, **config["optim"])

    if args.pre_training:
        print(f"[Saving pre-trained model weights to {args.pre_weights_path}]")
        save_model_weights(model, args.pre_weights_path)
    else:
        print(f"[Saving trained model weights to {args.weights_path}]")
        save_model_weights(model, args.weights_path)


def get_layer_weights(weights, prefix):
    layer_weights = OrderedDict()
    for key, value in weights.items():
        if key.startswith(prefix):
            layer_weights[key[len(prefix) :]] = value
    return layer_weights


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    # Pre-training arguments
    parser.add_argument(
        "--pre-training",
        action="store_true",
        help="Run pre-training before training the model",
    )
    pretraining_args = parser.add_argument_group("Pre-training arguments")
    pretraining_args.add_argument(
        "--pre-input-fasta", type=str, help="Path to input FASTA file for pre-training"
    )
    pretraining_args.add_argument(
        "--pre-input-fast5-dir",
        type=str,
        help="Path to directory of input FAST5 files for pre-training",
    )
    pretraining_args.add_argument(
        "--pre-chunksize", type=int, default=1000, help="Chunk size for pre-training"
    )
    pretraining_args.add_argument(
        "--pre-overlap", type=int, default=100, help="Overlap size for pre-training"
    )
    pretraining_args.add_argument(
        "--pre-max-homopolymer-length",
        type=int,
        default=3,
        help="Maximum homopolymer length for pre-training",
    )
    pretraining_args.add_argument(
        "--pre-weights-path",
        type=str,
        help="Path to save pre-trained model weights",
    )

    parser.add_argument(
        "--output-directory",
        type=str,
        help="Directory to save training data and checkpoints",
    )
    parser.add_argument(
        "--training-directory",
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
