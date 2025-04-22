#!/bin/bash
# Performs basecalling on the generated squiggles using the baseline Bonito model

MODEL=bonito/models/dna_r10.4.1_e8.2_400bps_sup@v5.0.0/
DATA_PATH=../data
OUTPUT_FILENAME=baseline_bonito.fastq

source /vol/bitbucket/sa2021/miniconda3/etc/profile.d/conda.sh

conda activate bonito-env
cd bonito
bonito basecaller $MODEL $DATA_PATH > $DATA_PATH/$OUTPUT_FILENAME
conda deactivate