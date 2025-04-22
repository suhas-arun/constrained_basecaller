#!/bin/bash
# Generates mock POD5 data for comparing basecalling models.

NUM_SEQUENCES=100000
SEQUENCE_LENGTH=500
MAX_HOMOPOLYMER_LENGTH=3
MIN_GC=0.4
MAX_GC=0.6
FASTA_FILE="data/mock_data.fasta"
POD5_FILE="data/mock_data.pod5"

source /vol/bitbucket/sa2021/miniconda3/etc/profile.d/conda.sh

# Generate FASTA file
conda activate bonito-env
python3 src/data_generator.py \
    --num_sequences $NUM_SEQUENCES \
    --sequence_length $SEQUENCE_LENGTH \
    --max_homopolymer_length $MAX_HOMOPOLYMER_LENGTH \
    --min_gc $MIN_GC \
    --max_gc $MAX_GC \
    --output_file $FASTA_FILE

# Convert sequences to squiggles (POD5)
conda activate seq2squiggle-env
seq2squiggle predict $FASTA_FILE -o $POD5_FILE --read-input
conda deactivate
