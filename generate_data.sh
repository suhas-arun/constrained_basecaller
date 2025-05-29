#!/bin/bash
# Generate mock FAST5 data for basecalling

##################################################

# File paths
ROOT_DIR=$(pwd)
FASTA_FILE=$1
BLOW5_FILE="$ROOT_DIR/data/reads.blow5"
BLOW5_DIR="$ROOT_DIR/data/blow5"
FAST5_DIR=$2

# Squigulator parameters
SQUIGULATOR_VERSION="v0.4.0"
SQUIGULATOR_DIR="$ROOT_DIR/lib/squigulator-$SQUIGULATOR_VERSION"
DNA_PROFILE="dna-r10-prom"

# Slow5tools parameters
SLOW5TOOLS_VERSION="v1.3.0"
SLOW5TOOLS_DIR="$ROOT_DIR/lib/slow5tools-$SLOW5TOOLS_VERSION"
NUM_THREADS=8

# Data generation parameters
NUM_SEQUENCES=$3
SEQUENCE_LENGTH=2000
CONSTRAINED_FLAG=$4
MAX_HOMOPOLYMER_LENGTH=3
MIN_GC=0.4
MAX_GC=0.6
MIN_HP_INSERT_LENGTH=3
MAX_HP_INSERT_LENGTH=5
HP_INSERT_PROBABILITY=0.1

##################################################

# Generate FASTA file
python3 src/data_generator.py \
    --num_sequences $NUM_SEQUENCES \
    --sequence_length $SEQUENCE_LENGTH \
    --max_homopolymer_length $MAX_HOMOPOLYMER_LENGTH \
    --min_gc $MIN_GC \
    --max_gc $MAX_GC \
    --output_file $FASTA_FILE \
    --min_hp_insert_length $MIN_HP_INSERT_LENGTH \
    --max_hp_insert_length $MAX_HP_INSERT_LENGTH \
    --hp_insert_probability $HP_INSERT_PROBABILITY \
    $CONSTRAINED_FLAG

# Convert sequences to squiggles (BLOW5)
$SQUIGULATOR_DIR/squigulator \
    $FASTA_FILE \
    -n $NUM_SEQUENCES \
    -r $SEQUENCE_LENGTH \
    -x $DNA_PROFILE \
    -o $BLOW5_FILE

# Split BLOW5 file for parallel processing
$SLOW5TOOLS_DIR/slow5tools split \
    $BLOW5_FILE \
    -d $BLOW5_DIR \
    -r $((NUM_SEQUENCES / NUM_THREADS + 1))

# Convert BLOW5 to FAST5
$SLOW5TOOLS_DIR/slow5tools s2f $BLOW5_DIR -d $FAST5_DIR

# Clean up
rm -rf $BLOW5_FILE
rm -rf $BLOW5_DIR
