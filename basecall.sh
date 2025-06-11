#!/bin/bash
# Generates mock data and performs basecalling

##################################################

# File paths
ROOT_DIR=$(pwd)
TRAIN_DIR="$ROOT_DIR/data/baseline/train"
TRAIN_FASTA_FILE="$TRAIN_DIR/mock_data.fasta"
TRAIN_FAST5_DIR="$TRAIN_DIR/fast5"

TEST_DIR="$ROOT_DIR/data/baseline/test"
TEST_FASTA_FILE="$TEST_DIR/mock_data.fasta"
TEST_FAST5_DIR="$TEST_DIR/fast5"

OUTPUT_DIR="$ROOT_DIR/out/baseline"
PRETRAINED_SAM_FILE="$OUTPUT_DIR/pretrained_basecalls.sam"
FINETUNED_OUTPUT_DIR="$OUTPUT_DIR/fine_tuned"
FINETUNED_SAM_FILE="$OUTPUT_DIR/finetuned_basecalls.sam"

NUM_TRAIN_SEQUENCES=40000
NUM_TEST_SEQUENCES=10000

# Baseline model parameters
MODEL="bonito/models/dna_r10.4.1_e8.2_400bps_sup@v5.0.0/"

##################################################

source /vol/bitbucket/sa2021/miniconda3/etc/profile.d/conda.sh
conda activate bonito-env

read -p "Do you want to generate new training and test data? (y/n): " confirm
if [[ $confirm == "y" || $confirm == "Y" ]]; then
    echo "--- Generating Training Data ($NUM_TRAIN_SEQUENCES sequences) ---"
    mkdir -p $TRAIN_DIR
    ./generate_data.sh $TRAIN_FASTA_FILE $TRAIN_FAST5_DIR $NUM_TRAIN_SEQUENCES --constrained

    echo "--- Generating Test Data ($NUM_TEST_SEQUENCES sequences) ---"
    mkdir -p $TEST_DIR
    ./generate_data.sh $TEST_FASTA_FILE $TEST_FAST5_DIR $NUM_TEST_SEQUENCES --constrained
else
    echo "Using previously generated training and test data."
fi

cd bonito

read -p "Do you want to fine-tune the model? (y/n): " fine_tune
if [[ $fine_tune != "y" && $fine_tune != "Y" ]]; then
    mkdir -p $OUTPUT_DIR
    echo "Using pre-trained Bonito model."

    # Basecalling with pre-trained model
    bonito basecaller \
        --reference $TEST_FASTA_FILE \
        $MODEL $TEST_FAST5_DIR > $PRETRAINED_SAM_FILE

    # Analyse basecalling results
    echo "Analysing basecalling results..."
    python3 $ROOT_DIR/src/constraint_analysis.py --sam_file $PRETRAINED_SAM_FILE
else
    # Fine-tune the model
    TRAIN_CTC_DIR="$ROOT_DIR/data/fine_tune/train"
    TRAIN_CTC_DATA="$TRAIN_CTC_DIR/basecalls.sam"

    # Training hyperparameters
    EPOCHS=5
    CHUNKS=8000
    VALID_CHUNKS=2000
    BATCH_SIZE=16

    mkdir -p $TRAIN_CTC_DIR
    rm -rf $FINETUNED_OUTPUT_DIR

    echo "Fine-tuning the Bonito model on the training set..."

    # Prepare training data
    bonito basecaller \
        --reference $TRAIN_FASTA_FILE \
        --save-ctc \
        --min-accuracy-save-ctc 0.8 \
        $MODEL $TRAIN_FAST5_DIR > $TRAIN_CTC_DATA

    bonito train \
        --directory $TRAIN_CTC_DIR \
        --epochs $EPOCHS \
        --chunks $CHUNKS \
        --valid-chunks $VALID_CHUNKS \
        --batch $BATCH_SIZE \
        $FINETUNED_OUTPUT_DIR

    echo "Evaluating the fine-tuned model on the test set..."
    
    # Basecalling with fine-tuned model
    bonito basecaller \
        --reference $TEST_FASTA_FILE \
        $FINETUNED_OUTPUT_DIR $TEST_FAST5_DIR > $FINETUNED_SAM_FILE

    # Analyse basecalling results
    echo "Analysing basecalling results..."
    python3 $ROOT_DIR/src/constraint_analysis.py --sam_file $FINETUNED_SAM_FILE
fi

cd $ROOT_DIR
conda deactivate
