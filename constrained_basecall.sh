#!/bin/bash
# Perform basecalling with the constraint-aware basecaller model

##################################################

ROOT_DIR=$(pwd)
OUTPUT_DIR="$ROOT_DIR/out/constrained"
WEIGHTS_FILE="$OUTPUT_DIR/final_weights.tar"

# Baseline model parameters
PRETRAINED_MODEL="dna_r10.4.1_e8.2_400bps_sup@v5.0.0/"

##################################################

source /vol/bitbucket/sa2021/miniconda3/etc/profile.d/conda.sh
conda activate bonito-env
export PYTHONPATH="bonito:src"

read -p "Do you want to train the model? (y/n): " train
if [[ $train != "y" && $train != "Y" ]]; then
    echo "Using pre-trained model."
else
    TRAINING_DIR="$ROOT_DIR/data/train"
    TRAINING_DATA="$TRAINING_DIR/basecalls.sam"
    TRAINING_FASTA_FILE="$TRAINING_DIR/reference.fasta"
    TRAINING_FAST5_DIR="$TRAINING_DIR/fast5"

    # Training hyperparameters
    EPOCHS=10
    CHUNKS=4000
    BATCH_SIZE=16

    # Clean up previous training data
    rm -rf $TRAINING_DIR
    rm -rf $OUTPUT_DIR
    mkdir -p $TRAINING_DIR

    read -p "Do you want to generate new training data? (y/n): " confirm
    if [[ $confirm != "y" && $confirm != "Y" ]]; then
        echo "Using previously generated training data."
    else
        ./generate_data.sh $TRAINING_FASTA_FILE $TRAINING_FAST5_DIR 10000

        echo "Preparaing training data..."
        bonito basecaller \
            --reference $TRAINING_FASTA_FILE \
            --save-ctc \
            --min-accuracy-save-ctc 0.8 \
            $PRETRAINED_MODEL $TRAINING_FAST5_DIR > $TRAINING_DATA
    fi

    python3 -m basecaller.train \
        --training-directory $OUTPUT_DIR \
        --directory $TRAINING_DIR \
        --epochs $EPOCHS \
        --chunks $CHUNKS \
        --batch $BATCH_SIZE \
        --weights-path $WEIGHTS_FILE

    cd $ROOT_DIR
fi

TEST_DIR="$ROOT_DIR/data/test"
TEST_FASTA_FILE="$TEST_DIR/reference.fasta"
TEST_FAST5_DIR="$TEST_DIR/fast5"
SAM_FILE="$OUTPUT_DIR/basecalls.sam"

mkdir -p $TEST_DIR

read -p "Do you want to generate new test data? (y/n): " confirm
if [[ $confirm != "y" && $confirm != "Y" ]]; then
    echo "Using previously generated test data."
else
    ./generate_data.sh $TEST_FASTA_FILE $TEST_FAST5_DIR 10000
fi

read -p "Enter the path to the weights file: " WEIGHTS_FILE

python3 -m basecaller.main \
    $TEST_FAST5_DIR \
    $TEST_FASTA_FILE \
    --weights-path $WEIGHTS_FILE \
    > $SAM_FILE

# Analyse basecalling results
echo "Analysing basecalling results..."
python3 src/constraint_analysis.py --sam_file $SAM_FILE

conda deactivate
