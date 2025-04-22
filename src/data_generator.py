import random
import argparse
from utils import find_homopolymers, get_gc_content, write_fasta


def generate_sequence(length, gc_content, max_homopolymer_length):
    """
    Generates random DNA sequence of given length with specified GC content.
    The sequence will not contain homopolymers longer than `max_homopolymer_length`.
    """
    assert 0 <= gc_content <= 1, "GC content must be between 0 and 1"

    gc_target = int(gc_content * length)
    at_target = length - gc_target

    gc_bases = ["G", "C"]
    at_bases = ["A", "T"]

    sequence = []

    while len(sequence) < length:
        # Choose from GC or AT based on remaining quota
        gc_target_ratio = gc_target / (gc_target + at_target)
        if gc_target > 0 and (at_target == 0 or random.random() < gc_target_ratio):
            candidate_bases = gc_bases
        else:
            candidate_bases = at_bases

        # Filter out bases that would cause homopolymers
        if len(sequence) >= max_homopolymer_length:
            last_bases = sequence[-max_homopolymer_length:]
            if len(set(last_bases)) == 1:
                repeated_base = last_bases[-1]
                candidate_bases = [b for b in candidate_bases if b != repeated_base]

        base = random.choice(candidate_bases)
        sequence.append(base)

        if base in gc_bases:
            gc_target -= 1
        else:
            at_target -= 1

    return "".join(sequence)


def generate_sequences(
    num_sequences, length, max_homopolymer_length, min_gc_content, max_gc_content
):
    """
    Generates list of random DNA sequences.
    """
    sequences = []
    for _ in range(num_sequences):
        gc_percent = random.uniform(min_gc_content, max_gc_content)
        seq = generate_sequence(length, gc_percent, max_homopolymer_length)

        assert (
            find_homopolymers(seq, max_homopolymer_length) == []
        ), "Homopolymer length exceeded"

        assert (
            min_gc_content <= get_gc_content(seq) <= max_gc_content
        ), "GC content out of bounds"

        sequences.append(seq)

    return sequences

def argparser():
    parser = argparse.ArgumentParser(description="Generate synthetic DNA sequences.")
    parser.add_argument(
        "--num_sequences",
        type=int,
        default=100000,
        help="Number of sequences to generate",
    )
    parser.add_argument(
        "--sequence_length", type=int, default=500, help="Length of each DNA sequence"
    )
    parser.add_argument(
        "--max_homopolymer_length",
        type=int,
        default=3,
        help="Maximum allowed homopolymer length",
    )
    parser.add_argument(
        "--min_gc",
        type=float,
        default=0.4,
        help="Minimum GC content (between 0 and 1)",
    )
    parser.add_argument(
        "--max_gc",
        type=float,
        default=0.6,
        help="Maximum GC content (between 0 and 1)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="data/mock_data.fasta",
        help="Output FASTA file path",
    )
    return parser

if __name__ == "__main__":
    parser = argparser()
    args = parser.parse_args()

    print("Generating sequences...")
    sequences = generate_sequences(
        args.num_sequences,
        args.sequence_length,
        args.max_homopolymer_length,
        args.min_gc,
        args.max_gc,
    )

    print("Writing sequences to FASTA file...")
    write_fasta(sequences, args.output_file)
    print("Done.")
