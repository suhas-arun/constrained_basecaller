from bonito.multiprocessing import process_cancel
from bonito.reader import Reader
import numpy as np


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
        sequence = self.sequences[sequence_id]
        base_start = int(signal_chunk_start / self.signal_points_per_base)
        base_end = int(signal_chunk_end / self.signal_points_per_base)
        base_end = min(base_end, len(sequence))

        sequence_segment = sequence[base_start:base_end]

        if len(signal_chunk) != self.chunksize:
            print(
                f"Warning: Signal chunk for {read_id} (idx {idx}) has len {len(signal_chunk)}, expected {self.chunksize}."
            )

        hp_lengths, is_hp, hp_bases = self.generate_homopolymer_labels(sequence_segment)

        hp_labels = {
            "hp_lengths": hp_lengths,
            "is_hp": is_hp,
            "hp_bases": hp_bases,
        }

        # TODO: downsample hp_labels to match signal_chunk downsampling in model

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


if __name__ == "__main__":
    d = PretrainDataset("data/train/stage1/reference.fasta", "data/train/stage1/fast5")
    item = d[0]
