#!/bin/bash

CODE_DIR=$PWD
DATA_DIR="$CODE_DIR/data/samples"
MODEL_DIR="$CODE_DIR/trained_models/sol_edge_predictor"

CROSS_VAL=true
SEED=0
NUM_CORES=4

# Training dataset configuration
INSTANCE_TYPE="fctp"
COST_STRUCTURE="agarwal-aneja"
SIZE=15
MAX_QUANT=20
THETA=0.2
BALANCE_FACTOR=1.0
DATA_SPEC="${INSTANCE_TYPE}_${COST_STRUCTURE}_${SIZE}_${SIZE}_B${MAX_QUANT}_Theta${THETA}_BF${BALANCE_FACTOR}"

for MODEL_NAME in "linear_logreg" "edge_mlp"
do
    for FEATURES in "combined_raw_edge_features" "advanced_edge_features" "advanced_edge_features_plus_stat_features"
    do
        python $CODE_DIR/scripts/02_training_and_evaluation/01_train_sol_edge_predictor.py data_path=$DATA_DIR/$DATA_SPEC model=$MODEL_NAME model.features=$FEATURES out_dir=$MODEL_DIR/$DATA_SPEC/$MODEL_NAME seed=$SEED cross_validate=$CROSS_VAL
    done
done