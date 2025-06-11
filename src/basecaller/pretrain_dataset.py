from bonito.multiprocessing import process_cancel
from bonito.reader import Reader
import numpy as np
import torch


class PretrainDataset:
    """
    Dataset class used for pretraining the basecaller.
    """

    BASES = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}

    def __init__(
        self,
        fasta_file,
        fast5_dir,
        chunksize=2000,
        overlap=200,
        signal_points_per_base=10.0,
        max_homopolymer_length=3,
    ):
        self.fasta_file = fasta_file
        self.fast5_dir = fast5_dir
        self.chunksize = chunksize
        self.overlap = overlap
        self.signal_points_per_base = signal_points_per_base
        self.max_homopolymer_length = max_homopolymer_length
        self.reader = Reader(fast5_dir, recursive=True)

        self.sequences = self.read_fasta()
        print(f"Loaded {len(self.sequences)} sequences from {self.fasta_file}.")
        self.chunk_metadata = self.get_chunk_metadata()
        print(f"Generated metadata for {len(self.chunk_metadata)} signal chunks.")

    def read_fasta(self):
        """
        Read sequences from a FASTA file.
        """
        sequences = {}
        current_header = None
        current_sequence = []
        for line in open(self.fasta_file, "r"):
            line = line.strip()
            if line.startswith(">"):
                if current_header is not None:
                    sequences[current_header] = "".join(current_sequence)
                current_header = line[1:]
                current_sequence = []
            elif current_header is not None:
                current_sequence.append(line)
        if current_header is not None:
            sequences[current_header] = "".join(current_sequence)
        return sequences

    def get_chunk_metadata(self):
        """
        Generate metadata for signal chunks from the FAST5 files.
        """
        self.signals = {}
        chunk_metadata = []
        print("Reading FAST5 files and generating chunk metadata...")
        reads = self.reader.get_reads(
            self.fast5_dir,
            n_proc=8,
            cancel=process_cancel(),
        )
        for read in reads:
            read_id = read.read_id
            signal = read.signal
            signal_len = len(signal)
            self.signals[read_id] = signal

            for signal_start in range(0, signal_len - self.chunksize + 1, self.overlap):
                chunk_metadata.append(
                    {
                        "read_id": read_id,
                        "signal_chunk_start": signal_start,
                        "signal_chunk_end": signal_start + self.chunksize,
                    }
                )

        return chunk_metadata

    def __len__(self) -> int:
        return len(self.chunk_metadata)

    def __getitem__(self, idx: int):
        metadata = self.chunk_metadata[idx]
        read_id = metadata["read_id"]

        signal_chunk_start = metadata["signal_chunk_start"]
        signal_chunk_end = metadata["signal_chunk_end"]
        signal = self.signals[read_id]
        signal_chunk = signal[signal_chunk_start:signal_chunk_end]

        sequence_id = read_id.split("!")[1]
        full_sequence = self.sequences[sequence_id]
        base_start = int(signal_chunk_start / self.signal_points_per_base)
        base_end = int(signal_chunk_end / self.signal_points_per_base)
        base_end = min(base_end, len(full_sequence))

        sequence_segment = full_sequence[base_start:base_end]

        if len(signal_chunk) != self.chunksize:
            print(
                f"Warning: Signal chunk for {read_id} (idx {idx}) has len {len(signal_chunk)}, expected {self.chunksize}."
            )

        hp_lengths, is_hp, hp_bases = self.generate_homopolymer_labels(sequence_segment)
        target_hp_feature_length = self.get_hp_feature_length()

        hp_labels = self.downsample_hp_labels(
            hp_lengths, is_hp, hp_bases, len(sequence_segment), target_hp_feature_length
        )

        return signal_chunk, hp_labels

    def generate_homopolymer_labels(self, sequence):
        """
        Generate homopolymer labels for a given sequence.
        """
        n = len(sequence)
        hp_lengths = np.zeros(n, dtype=np.int32)
        is_hp = np.zeros(n, dtype=np.bool_)
        hp_bases = np.full(n, self.BASES["N"], dtype=np.int32)

        i = 0
        while i < n:
            current_base = sequence[i]
            j = i
            while j < n and sequence[j] == current_base:
                j += 1
            length = j - i
            if length >= self.max_homopolymer_length:
                hp_base_index = self.BASES.get(current_base, self.BASES["N"])
                for k in range(i, j):
                    hp_lengths[k] = length
                    is_hp[k] = True
                    hp_bases[k] = hp_base_index
            i = j

        return hp_lengths, is_hp, hp_bases

    def downsample_hp_labels(
        self, hp_lengths, is_hp, hp_bases, sequence_length, target_length
    ):
        """
        Downsample homopolymer labels to match the target length.
        """
        downsampled_hp_lengths = np.zeros(target_length, dtype=np.float32)
        downsampled_is_hp = np.zeros(target_length, dtype=np.bool_)
        downsampled_hp_bases = np.full(
            (target_length,), self.BASES["N"], dtype=np.int32
        )

        if sequence_length == 0:
            return {
                "hp_lengths": torch.from_numpy(downsampled_hp_lengths),
                "is_hp": torch.from_numpy(downsampled_is_hp),
                "hp_bases": torch.from_numpy(downsampled_hp_bases),
            }

        bases_per_hp_feature = sequence_length / target_length
        for i in range(target_length):
            start_index = int(i * bases_per_hp_feature)
            end_index = int((i + 1) * bases_per_hp_feature)
            end_index = min(end_index, len(hp_lengths))

            segment_hp_lengths = hp_lengths[start_index:end_index]
            segment_is_hp = is_hp[start_index:end_index]
            segment_hp_bases = hp_bases[start_index:end_index]

            if segment_is_hp.any():
                downsampled_is_hp[i] = True
                downsampled_hp_lengths[i] = segment_hp_lengths[segment_is_hp].max()
                # If there are multiple homopolymers, take the most frequent base
                if segment_hp_bases[segment_is_hp].size > 0:
                    # convert to tensor for bincount
                    segment_hp_bases = torch.from_numpy(segment_hp_bases[segment_is_hp])
                    downsampled_hp_bases[i] = (
                        torch.bincount(segment_hp_bases).argmax().item()
                    )
            else:
                downsampled_is_hp[i] = False

        return {
            "hp_lengths": torch.from_numpy(downsampled_hp_lengths),
            "is_hp": torch.from_numpy(downsampled_is_hp),
            "hp_bases": torch.from_numpy(downsampled_hp_bases),
        }

    def get_hp_feature_length(self):
        """
        Calculate the length of the homopolymer feature vector.
        """
        length = self.chunksize
        length = get_conv_output_length(length, 5, 1, 2)
        length = get_conv_output_length(length, 5, 1, 2)
        length = get_conv_output_length(length, 9, 3, 4)
        return length


def hp_collate_fn(batch):
    """
    Custom collate function to handle variable-length sequences and homopolymer labels.
    """
    signals, hp_labels = zip(*batch)
    batched_signals = np.array(signals)
    batched_signals = torch.tensor(batched_signals, dtype=torch.float32)

    batched_hp_lengths = torch.from_numpy(
        np.array([labels["hp_lengths"] for labels in hp_labels])
    )
    batched_is_hp = torch.from_numpy(
        np.array([labels["is_hp"] for labels in hp_labels])
    )
    batched_hp_bases = torch.from_numpy(
        np.array([labels["hp_bases"] for labels in hp_labels])
    )

    return batched_signals, batched_hp_lengths, batched_is_hp, batched_hp_bases


def get_conv_output_length(input_length, kernel_size, stride=1, padding=0):
    """
    Calculate the output length of a convolutional layer.
    """
    return (input_length + 2 * padding - kernel_size) // stride + 1


if __name__ == "__main__":
    d = PretrainDataset("data/train/stage1/reference.fasta", "data/train/stage1/fast5")
    item = d[0]
