#!/bin/bash
# Perform basecalling with the constraint-aware basecaller model

ROOT_DIR=$(pwd)
FASTA_FILE="$ROOT_DIR/data/reference.fasta"
FAST5_DIR="$ROOT_DIR/data/fast5"
SAM_FILE="$ROOT_DIR/out/basecalls.sam"

source /vol/bitbucket/sa2021/miniconda3/etc/profile.d/conda.sh
conda activate bonito-env

read -p "Do you want to generate new data? (y/n): " confirm
if [[ $confirm != "y" && $confirm != "Y" ]]; then
    echo "Using previously generated data."
else
    ./generate_data.sh $FASTA_FILE $FAST5_DIR
fi

PYTHONPATH="bonito:src" \
    python3 -m basecaller.main \
        $FAST5_DIR \
        $FASTA_FILE \
        > $SAM_FILE

# Analyse basecalling results
echo "Analysing basecalling results..."
python3 src/constraint_analysis.py --sam_file $SAM_FILE

conda deactivate
