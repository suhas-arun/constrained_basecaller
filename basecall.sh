#!/bin/bash
# Generates mock data and performs basecalling

##################################################

# Data generation parameters
NUM_SEQUENCES=100000
SEQUENCE_LENGTH=500
MAX_HOMOPOLYMER_LENGTH=3
MIN_GC=0.4
MAX_GC=0.6

# File paths
ROOT_DIR=$(pwd)
FASTA_FILE="$ROOT_DIR/data/mock_data.fasta"
BLOW5_FILE="$ROOT_DIR/data/reads.blow5"
BLOW5_DIR="$ROOT_DIR/data/blow5"
FAST5_DIR="$ROOT_DIR/data/fast5"
SAM_FILE="$ROOT_DIR/out/basecalls.sam"

# Squigulator parameters
SQUIGULATOR_VERSION="v0.4.0"
SQUIGULATOR_DIR="$ROOT_DIR/lib/squigulator-$SQUIGULATOR_VERSION"
DNA_PROFILE="dna-r10-prom"

# Slow5tools parameters
SLOW5TOOLS_VERSION="v1.3.0"
SLOW5TOOLS_DIR="$ROOT_DIR/lib/slow5tools-$SLOW5TOOLS_VERSION"
NUM_THREADS=8

# Baseline model parameters
MODEL="bonito/models/dna_r10.4.1_e8.2_400bps_sup@v5.0.0/"

##################################################

source /vol/bitbucket/sa2021/miniconda3/etc/profile.d/conda.sh

# Clean up previous runs
rm -rf "$ROOT_DIR/data"
rm -rf "$ROOT_DIR/out"
mkdir -p "$ROOT_DIR/data"
mkdir -p "$ROOT_DIR/out"

# Generate FASTA file
conda activate bonito-env
python3 src/data_generator.py \
    --num_sequences $NUM_SEQUENCES \
    --sequence_length $SEQUENCE_LENGTH \
    --max_homopolymer_length $MAX_HOMOPOLYMER_LENGTH \
    --min_gc $MIN_GC \
    --max_gc $MAX_GC \
    --output_file $FASTA_FILE

# Convert sequences to squiggles (BLOW5)
$SQUIGULATOR_DIR/squigulator \
    $FASTA_FILE \
    -n $NUM_SEQUENCES \
    -r $SEQUENCE_LENGTH \
    -x $DNA_PROFILE \
    -o $BLOW5_FILE \

# Split BLOW5 file for parallel processing
$SLOW5TOOLS_DIR/slow5tools split \
    $BLOW5_FILE \
    -d $BLOW5_DIR \
    -r $((NUM_SEQUENCES / NUM_THREADS + 1))
    
# Convert BLOW5 to FAST5
$SLOW5TOOLS_DIR/slow5tools s2f $BLOW5_DIR -d $FAST5_DIR

# Perform basecalling using Bonito
cd bonito
bonito basecaller \
    --reference $FASTA_FILE \
    $MODEL $FAST5_DIR > $SAM_FILE

# Clean up
rm -rf $BLOW5_FILE
rm -rf $BLOW5_DIR
rm -rf $FAST5_DIR

conda deactivate