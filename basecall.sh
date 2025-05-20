#!/bin/bash
# Generates mock data and performs basecalling

##################################################

# File paths
ROOT_DIR=$(pwd)
FASTA_FILE="$ROOT_DIR/data/mock_data.fasta"
FAST5_DIR="$ROOT_DIR/data/fast5"
SAM_FILE="$ROOT_DIR/out/basecalls.sam"

# Baseline model parameters
MODEL="bonito/models/dna_r10.4.1_e8.2_400bps_sup@v5.0.0/"

##################################################

source /vol/bitbucket/sa2021/miniconda3/etc/profile.d/conda.sh
conda activate bonito-env

read -p "Do you want to generate new data? (y/n): " confirm
if [[ $confirm != "y" && $confirm != "Y" ]]; then
    echo "Using previously generated data."
else
    ./generate_data.sh $FASTA_FILE $FAST5_DIR 10000
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
    EPOCHS=20
    CHUNKS=400
    VALID_CHUNKS=20
    BATCH_SIZE=16

    mkdir -p $TRAINING_DIR
    rm -rf $OUTPUT_DIR

    echo "Fine-tuning the Bonito model..."

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
