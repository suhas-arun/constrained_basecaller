import random
import argparse
from utils import find_homopolymers, get_gc_content, write_fasta


def generate_constrained_sequence(length, gc_content, max_homopolymer_length):
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


def generate_unconstrained_sequence(
    length, min_hp_insert_length, max_hp_insert_length, hp_insert_probability
):
    """
    Generates a random DNA sequence of given length containing homopolymers.
    The homopolymers will be between `min_hp_insert_length` and `max_hp_insert_length`.
    The probability of a homopolymer occurring is controlled by `hp_insert_probability`.
    """
    assert (
        min_hp_insert_length <= max_hp_insert_length
    ), "Minimum homopolymer length must be less than or equal to maximum"

    assert (
        0 <= hp_insert_probability <= 1
    ), "Homopolymer probability must be between 0 and 1"

    bases = ["A", "T", "G", "C"]
    sequence = []

    while len(sequence) < length:
        base = random.choice(bases)
        if random.random() < hp_insert_probability:
            # Generate a homopolymer
            remaining_length = length - len(sequence)
            homopolymer_length = min(
                remaining_length,
                random.randint(min_hp_insert_length, max_hp_insert_length),
            )
            sequence.extend([base] * homopolymer_length)
        else:
            # Add a single base
            sequence.append(base)


    return "".join(sequence)


def generate_constrained_sequences(
    num_sequences,
    length,
    max_homopolymer_length=None,
    min_gc_content=None,
    max_gc_content=None,
):
    """
    Generates list of random DNA sequences.
    """
    sequences = []
    for _ in range(num_sequences):
        gc_percent = random.uniform(min_gc_content, max_gc_content)
        seq = generate_constrained_sequence(length, gc_percent, max_homopolymer_length)

        assert (
            find_homopolymers(seq, max_homopolymer_length) == []
        ), "Homopolymer length exceeded"

        assert (
            min_gc_content <= get_gc_content(seq) <= max_gc_content
        ), "GC content out of bounds"

        sequences.append(seq)

    return sequences


def generate_unconstrained_sequences(
    num_sequences,
    length,
    min_hp_insert_length=3,
    max_hp_insert_length=5,
    hp_insert_probability=0.1,
):
    """
    Generates a list of unconstrained random DNA sequences.
    """
    sequences = []
    for _ in range(num_sequences):
        seq = generate_unconstrained_sequence(
            length,
            min_hp_insert_length=min_hp_insert_length,
            max_hp_insert_length=max_hp_insert_length,
            hp_insert_probability=hp_insert_probability,
        )
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
    # Constrained generation parameters
    parser.add_argument(
        "--constrained",
        action="store_true",
        help="Generate constrained sequences with specified GC content and homopolymer length",
    )
    parser.add_argument(
        "--max_homopolymer_length",
        type=int,
        default=3,
        help="Maximum allowed homopolymer length (for constrained sequences)",
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
    # Unconstrained generation parameters
    parser.add_argument(
        "--min_hp_insert_length",
        type=int,
        default=3,
        help="Minimum homopolymer length for unconstrained sequences",
    )
    parser.add_argument(
        "--max_hp_insert_length",
        type=int,
        default=5,
        help="Maximum homopolymer length for unconstrained sequences",
    )
    parser.add_argument(
        "--hp_insert_probability",
        type=float,
        default=0.1,
        help="Probability of inserting a homopolymer in unconstrained sequences",
    )
    # Output file
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
    if args.constrained:
        sequences = generate_constrained_sequences(
            args.num_sequences,
            args.sequence_length,
            args.max_homopolymer_length,
            args.min_gc,
            args.max_gc,
        )
    else:
        sequences = generate_unconstrained_sequences(
            args.num_sequences,
            args.sequence_length,
            args.min_hp_insert_length,
            args.max_hp_insert_length,
            args.hp_insert_probability,
        )

    print("Writing sequences to FASTA file:", args.output_file)
    write_fasta(sequences, args.output_file)
    print("Done.")
