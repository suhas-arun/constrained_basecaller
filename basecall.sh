#!/bin/bash
# Generates mock data and performs basecalling

##################################################

# Data generation parameters
NUM_SEQUENCES=1000
SEQUENCE_LENGTH=2000
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
conda activate bonito-env

read -p "Do you want to generate new data? (y/n): " confirm
if [[ $confirm != "y" && $confirm != "Y" ]]; then
    echo "Using previously generated data."
else
    # Clean up previous runs
    rm -rf "$ROOT_DIR/data"
    rm -rf "$ROOT_DIR/out"
    mkdir -p "$ROOT_DIR/data"
    mkdir -p "$ROOT_DIR/out"

    # Generate FASTA file
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
fi

read -p "Do you want to use Bonito for basecalling? (y/n): " bonito
if [[ $bonito != "y" && $bonito != "Y" ]]; then
    echo "Skipping basecalling."
    exit 0
fi

cd bonito

read -p "Do you want to fine-tune the model? (y/n): " fine_tune
if [[ $fine_tune != "y" && $fine_tune != "Y" ]]; then
    echo "Using pre-trained Bonito model."

    # Basecalling with pre-trained model
    bonito basecaller \
        --reference $FASTA_FILE \
        $MODEL $FAST5_DIR > $SAM_FILE
    cd $ROOT_DIR
else
    # Fine-tune the model
    TRAINING_DIR="$ROOT_DIR/data/train"
    TRAINING_DATA="$TRAINING_DIR/basecalls.sam"
    OUTPUT_DIR="$ROOT_DIR/out/fine_tuned"

    # Training hyperparameters
    EPOCHS=10
    CHUNKS=400
    VALID_CHUNKS=10
    BATCH_SIZE=16

    mkdir -p $TRAINING_DIR
    rm -rf $OUTPUT_DIR

    # Prepare training data
    bonito basecaller \
        --reference $FASTA_FILE \
        --save-ctc \
        --min-accuracy-save-ctc 0.8 \
        $MODEL $FAST5_DIR > $TRAINING_DATA

    bonito train \
        --directory $TRAINING_DIR \
        --epochs $EPOCHS \
        --chunks $CHUNKS \
        --valid-chunks $VALID_CHUNKS \
        --batch $BATCH_SIZE \
        $OUTPUT_DIR
    
    # Basecalling with fine-tuned model
    bonito basecaller \
        --reference $FASTA_FILE \
        $OUTPUT_DIR $FAST5_DIR > $SAM_FILE
    cd $ROOT_DIR
fi

# Analyse basecalling results
echo "Analysing basecalling results..."
python3 src/constraint_analysis.py --sam_file $SAM_FILE

conda deactivate
