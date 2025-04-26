import argparse
from utils import find_homopolymers, get_gc_content


def get_sequences_from_sam_file(filename):
    """
    Extract basecalled sequences from a SAM file.
    """
    sequences = []
    with open(filename, "r") as f:
        for line in f:
            if line.startswith("@"):
                continue
            fields = line.strip().split("\t")
            seq = fields[9]
            sequences.append(seq)
    return sequences


def analyse_sequences(sequences, max_homopolymer_length=3, min_gc=0.4, max_gc=0.6):
    """
    Analyse sequences for homopolymers and GC content.
    """
    homopolymers = []
    low_gc = []
    high_gc = []
    for seq in sequences:
        if find_homopolymers(seq, max_homopolymer_length):
            homopolymers.append(seq)

        gc_content = get_gc_content(seq)
        if gc_content < min_gc:
            low_gc.append(seq)

        if gc_content > max_gc:
            high_gc.append(seq)

    num_sequences = len(sequences)

    print_stat(len(homopolymers), num_sequences, "homopolymers")
    print_stat(len(low_gc), num_sequences, "low GC content")
    print_stat(len(high_gc), num_sequences, "high GC content")
    print_stat(
        num_sequences - len(low_gc) - len(high_gc), num_sequences, "normal GC content"
    )


def print_stat(stat_count, num_sequences, stat_text):
    print(
        f"{stat_count} / {num_sequences} ({round(stat_count / num_sequences * 100, 2)}%) "
        + f"sequences have {stat_text}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse sequences from a SAM file.")
    parser.add_argument(
        "--sam_file",
        type=str,
        help="Path to the SAM file",
    )
    args = parser.parse_args()

    sequences = get_sequences_from_sam_file(args.sam_file)
    analyse_sequences(sequences)
