import os
import torch


def find_homopolymers(sequence, max_length=3):
    """
    Find homopolymers greater than length `max_length` in the sequence.
    """
    homopolymers = []
    current_base = None
    current_count = 0

    for base in sequence:
        if base == current_base:
            current_count += 1
        else:
            if current_count > max_length:
                homopolymers.append((current_base, current_count))
            current_base = base
            current_count = 1

    if current_count > max_length:
        homopolymers.append((current_base, current_count))

    return homopolymers


def get_gc_content(sequence):
    """
    Calculate the GC content of a sequence.
    """
    gc_count = sequence.count("G") + sequence.count("C")
    return gc_count / len(sequence) if len(sequence) > 0 else 0.0


def write_fasta(sequences, filename, line_length=80):
    """
    Write a list of DNA sequences to a FASTA file.
    """
    with open(filename, "w") as f:
        for i, seq in enumerate(sequences):
            f.write(f">sequence_{i}\n")
            for j in range(0, len(seq), line_length):
                f.write(seq[j : j + line_length] + "\n")


def load_model_weights(model, weights_path):
    """
    Load model weights from a specified path.
    """
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path))
    else:
        raise FileNotFoundError(f"Model weights not found at {weights_path}")


def save_model_weights(model, weights_path):
    """
    Save model weights to a specified path.
    """
    torch.save(model.state_dict(), weights_path)
