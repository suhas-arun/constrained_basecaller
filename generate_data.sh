#!/bin/bash

source /vol/bitbucket/sa2021/miniconda3/etc/profile.d/conda.sh

# Generate FASTA file
conda activate bonito-env
python3 src/data_generator.py

# Convert sequences to squiggles (POD5)
conda activate seq2squiggle-env
seq2squiggle predict data/mock_data.fasta -o data/mock_data.pod5 --read-input
conda deactivate
