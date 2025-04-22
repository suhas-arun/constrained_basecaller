import random
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


if __name__ == "__main__":
    NUM_SEQUENCES = 100000
    SEQUENCE_LENGTH = 500
    MAX_HOMOPOLYMER_LENGTH = 3
    MIN_GC_CONTENT = 0.4
    MAX_GC_CONTENT = 0.6

    print("Generating sequences...")
    sequences = generate_sequences(
        NUM_SEQUENCES,
        SEQUENCE_LENGTH,
        MAX_HOMOPOLYMER_LENGTH,
        MIN_GC_CONTENT,
        MAX_GC_CONTENT,
    )

    print("Writing sequences to FASTA file...")
    write_fasta(sequences, "../data/mock_data.fasta")
    print("Done.")
