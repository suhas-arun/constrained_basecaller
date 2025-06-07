#!/bin/bash
# Perform basecalling with the constraint-aware basecaller model

##################################################

ROOT_DIR=$(pwd)

STAGE1_TRAINING_DIR="$ROOT_DIR/data/train/stage1"
STAGE1_FASTA_FILE="$STAGE1_TRAINING_DIR/reference.fasta"
STAGE1_FAST5_DIR="$STAGE1_TRAINING_DIR/fast5"
STAGE1_OUTPUT_DIR="$ROOT_DIR/out/stage1"
STAGE1_WEIGHTS_FILE="$STAGE1_OUTPUT_DIR/weights.tar"

STAGE2_TRAINING_DIR="$ROOT_DIR/data/train/stage2"
STAGE2_OUTPUT_DIR="$ROOT_DIR/out/stage2"
STAGE2_WEIGHTS_FILE="$STAGE2_OUTPUT_DIR/weights.tar"

# Baseline model parameters
PRETRAINED_MODEL="dna_r10.4.1_e8.2_400bps_sup@v5.0.0/"

##################################################

source /vol/bitbucket/sa2021/miniconda3/etc/profile.d/conda.sh
conda activate bonito-env
export PYTHONPATH="bonito:src"

# STAGE 1: Homopolymer Extractor Pre-training

read -p "Do you want to perform Stage 1: Homopolymer Extractor Pre-training (y/n): " stage1
if [[ $stage1 == "y" || $stage1 == "Y" ]]; then
    read -p "Do you want to generate new unconstrained training data? (y/n): " confirm
    if [[ $confirm != "y" && $confirm != "Y" ]]; then
        echo "Using previously generated Stage 1 data."
    else
        # Clean up previous training data
        rm -rf $STAGE1_TRAINING_DIR
        mkdir -p $STAGE1_TRAINING_DIR
        
        ./generate_data.sh \
            $STAGE1_FASTA_FILE \
            $STAGE1_FAST5_DIR \
            10000
    fi

    python3 -m basecaller.train \
        --pre-training \
        --pre-input-fasta $STAGE1_FASTA_FILE \
        --pre-input-fast5 $STAGE1_FAST5_DIR \
        --output-directory $STAGE1_OUTPUT_DIR \
        --training-directory $STAGE1_TRAINING_DIR \
        --pre-weights-path $STAGE1_WEIGHTS_FILE \
        --epochs 5 \
        --chunks 4000 \
        --batch 16 \

else
    # read -p "Skipping pre-training. Enter the path to the pre-trained weights file (STAGE 1): " STAGE1_WEIGHTS_FILE
    STAGE1_WEIGHTS_FILE="$STAGE1_OUTPUT_DIR/weights_5.tar"
fi

read -p "Do you want to perform Stage 2: Constraint-aware Basecaller Training (y/n): " stage2
if [[ $stage2 != "y" && $stage2 != "Y" ]]; then
    read -p "Skipping training. Enter the path to the weights file: " STAGE2_WEIGHTS_FILE
else
    STAGE2_TRAINING_DIR="$ROOT_DIR/data/train/stage2"
    STAGE2_TRAINING_DATA="$STAGE2_TRAINING_DIR/basecalls.sam"
    STAGE2_TRAINING_FASTA_FILE="$STAGE2_TRAINING_DIR/reference.fasta"
    STAGE2_TRAINING_FAST5_DIR="$STAGE2_TRAINING_DIR/fast5"

    # Training hyperparameters
    EPOCHS=10
    CHUNKS=4000
    BATCH_SIZE=16

    read -p "Do you want to generate new constrained training data? (y/n): " confirm
    if [[ $confirm != "y" && $confirm != "Y" ]]; then
        echo "Using previously generated training data."
    else
        # Clean up previous training data
        rm -rf $STAGE2_TRAINING_DIR
        rm -rf $STAGE2_OUTPUT_DIR
        mkdir -p $STAGE2_TRAINING_DIR

        ./generate_data.sh \
            $STAGE2_TRAINING_FASTA_FILE \
            $STAGE2_TRAINING_FAST5_DIR \
            10000 \
            --constrained

        echo "Preparaing training data..."
        bonito basecaller \
            --reference $STAGE2_TRAINING_FASTA_FILE \
            --save-ctc \
            --min-accuracy-save-ctc 0.8 \
            $PRETRAINED_MODEL $STAGE2_TRAINING_FAST5_DIR > $STAGE2_TRAINING_DATA
    fi

    # TODO: load stage 1 + 2 weights if available

    if [[ ! -f $STAGE1_WEIGHTS_FILE ]]; then
        echo "Pre-trained weights file not found: $STAGE1_WEIGHTS_FILE"
    else
        python3 -m basecaller.train \
            --output-directory $STAGE2_OUTPUT_DIR \
            --training-directory $STAGE2_TRAINING_DIR \
            --epochs $EPOCHS \
            --chunks $CHUNKS \
            --batch $BATCH_SIZE \
            --pre-weights-path $STAGE1_WEIGHTS_FILE \
            --weights-path $STAGE2_WEIGHTS_FILE
    fi
    cd $ROOT_DIR
fi

TEST_DIR="$ROOT_DIR/data/test"
TEST_FASTA_FILE="$TEST_DIR/reference.fasta"
TEST_FAST5_DIR="$TEST_DIR/fast5"
SAM_FILE="$STAGE2_OUTPUT_DIR/basecalls.sam"

mkdir -p $TEST_DIR

read -p "Do you want to generate new test data? (y/n): " confirm
if [[ $confirm != "y" && $confirm != "Y" ]]; then
    echo "Using previously generated test data."
else
    ./generate_data.sh \
        $TEST_FASTA_FILE \
        $TEST_FAST5_DIR \
        10000 \
        --constrained
fi

python3 -m basecaller.main \
    $TEST_FAST5_DIR \
    $TEST_FASTA_FILE \
    --weights-path $STAGE2_WEIGHTS_FILE \
    > $SAM_FILE

# Analyse basecalling results
echo "Analysing basecalling results..."
python3 src/constraint_analysis.py --sam_file $SAM_FILE

conda deactivate
