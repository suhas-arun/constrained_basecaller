import torch
from tqdm import tqdm
from time import perf_counter
import sys

from basecaller.model import ConstraintAwareBasecaller
from basecaller.basecall import basecall

from bonito.reader import Reader
from bonito.aligner import Aligner, align_map
from bonito.multiprocessing import process_cancel
from bonito.io import Writer
from bonito.util import tqdm_environ


def main(args):
    try:
        reader = Reader(args.reads_directory, recursive=True)
        sys.stderr.write(f"> reading {reader.fmt}")
    except FileNotFoundError:
        sys.stderr.write(
            f"> error: no suitable files found in {args.reads_directory}\n"
        )
        exit(1)

    model = ConstraintAwareBasecaller()
    if args.weights_path:
        sys.stderr.write(f"> loading weights from {args.weights_path}\n")
        model.load_state_dict(torch.load(args.weights_path))
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).half()
    model.eval()

    chunksize = args.chunksize - args.chunksize % model.stride
    overlap = args.overlap - args.overlap % (model.stride * 2)
    model.use_koi(
        batchsize=args.batchsize,
        chunksize=chunksize,
    )

    sys.stderr.write(f"> loading reference {args.reference_file}\n")
    aligner = Aligner(args.reference_file, preset="lr:hq")

    groups, num_reads = reader.get_read_groups(
        args.reads_directory,
        None,
        n_proc=8,
        cancel=process_cancel(),
    )

    reads = reader.get_reads(
        args.reads_directory,
        n_proc=8,
        cancel=process_cancel(),
    )

    results = basecall(
        model,
        reads,
        batchsize=args.batchsize,
        chunksize=chunksize,
        overlap=overlap,
    )

    aligned_results = align_map(aligner, results, n_thread=8)

    writer_kwargs = {
        "aligner": aligner,
        "ref_fn": args.reference_file,
        "groups": groups,
    }

    writer = Writer(
        "w",
        tqdm(
            aligned_results,
            desc="> calling",
            unit=" reads",
            total=num_reads,
            ncols=100,
            **tqdm_environ(),
        ),
        **writer_kwargs,
    )

    t0 = perf_counter()
    writer.start()
    writer.join()
    duration = perf_counter() - t0
    num_samples = sum(num_samples for _, num_samples in writer.log)

    sys.stderr.write(f"> completed read: {len(writer.log)}\n")
    sys.stderr.write(f"> duration: {duration:.2f} seconds\n")
    sys.stderr.write(f"> samples per second {num_samples / duration:.2f}\n")
    sys.stderr.write("> done\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Constraint-aware basecaller for reading synthetic data."
    )
    parser.add_argument("reads_directory", type=str, help="Directory containing reads")
    parser.add_argument(
        "reference_file", type=str, help="Reference file for alignment (FASTA format)"
    )
    parser.add_argument(
        "--batchsize",
        type=int,
        default=64,
        help="Batch size for basecalling (default: 64)",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=4000,
        help="Chunk size for basecalling (default: 4000)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=500,
        help="Overlap size for basecalling (default: 500)",
    )
    parser.add_argument(
        "--weights-path",
        type=str,
        help="Path to weights file (default: None)",
    )

    args = parser.parse_args()
    main(args)
