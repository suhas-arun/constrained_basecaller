#!/bin/bash
# Perform basecalling with the constraint-aware basecaller model

##################################################

ROOT_DIR=$(pwd)
FASTA_FILE="$ROOT_DIR/data/reference.fasta"
FAST5_DIR="$ROOT_DIR/data/fast5"
SAM_FILE="$ROOT_DIR/out/basecalls.sam"

TRAINING_DIR="$ROOT_DIR/data/train"
TRAINING_DATA="$TRAINING_DIR/basecalls.sam"
OUTPUT_DIR="$ROOT_DIR/out/constrained"
WEIGHTS_FILE="$OUTPUT_DIR/final_weights.tar"

# Baseline model parameters
MODEL="dna_r10.4.1_e8.2_400bps_sup@v5.0.0/"

##################################################

source /vol/bitbucket/sa2021/miniconda3/etc/profile.d/conda.sh
conda activate bonito-env
export PYTHONPATH="bonito:src"

read -p "Do you want to generate new data? (y/n): " confirm
if [[ $confirm != "y" && $confirm != "Y" ]]; then
    echo "Using previously generated data."
else
    ./generate_data.sh $FASTA_FILE $FAST5_DIR
fi

read -p "Do you want to train the model? (y/n): " train
if [[ $train != "y" && $train != "Y" ]]; then
    echo "Using pre-trained model."
else
    # Training hyperparameters
    EPOCHS=20
    CHUNKS=4000
    BATCH_SIZE=16

    mkdir -p $TRAINING_DIR
    rm -rf $OUTPUT_DIR

    echo "Preparaing training data..."
    bonito basecaller \
        --reference $FASTA_FILE \
        --save-ct \
        --min-accuracy-save-ctc 0.8 \
        $MODEL $FAST5_DIR > $TRAINING_DATA

    python3 -m basecaller.train \
        --training-directory $OUTPUT_DIR \
        --directory $TRAINING_DIR \
        --epochs $EPOCHS \
        --chunks $CHUNKS \
        --batch $BATCH_SIZE \
        --weights-path $WEIGHTS_FILE

    cd $ROOT_DIR
fi

python3 -m src/basecaller.main \
    $FAST5_DIR \
    $FASTA_FILE \
    --weights-path $WEIGHTS_FILE \
    > $SAM_FILE

# Analyse basecalling results
echo "Analysing basecalling results..."
python3 src/constraint_analysis.py --sam_file $SAM_FILE

conda deactivate
