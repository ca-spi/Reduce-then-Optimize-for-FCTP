#!/bin/bash

CODE_DIR=$PWD
DATA_DIR="$CODE_DIR/data/instances/benchmarking"
SOLUTION_DIR="$CODE_DIR/benchmarking"
MODEL_DIR="$CODE_DIR/trained_models/sol_edge_predictor"

SEED=0
NUM_CORES=8
declare -A TIME_LIMIT
TIME_LIMIT[30]=43200
TIME_LIMIT[120]=7200

# Dataset configuration (adjust to evaluate different dataset)
INSTANCE_TYPE="fctp"
COST_STRUCTURE="agarwal-aneja"
MAX_QUANT=20
THETA=0.2
BALANCE_FACTOR=1.0
TRAIN_SIZE=15

# Model path and name (adjust to evaluate different model)
MODEL_SPEC="gcnn/model_gcnn_features_bipartite_raw_prediction_task_binary_classification_normalization_standard_hidden_layer_dim_20_num_conv_layers_10_num_dense_layers_2/cross_val/best_checkpoint.pth.tar"
MODEL_NAME="model_gcnn_features_bipartite_raw_prediction_task_binary_classification_normalization_standard_hidden_layer_dim_20_num_conv_layers_10_num_dense_layers_2"

# Edge set size thresholds
declare -A THRSH
THRSH[30]="0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.6 0.8 1.0"
THRSH[120]="0.025 0.05 0.1 0.2 0.3 0.4"

for SIZE in 30 120
do
    IFS=' ' read -r -a THRSH_VALS <<< ${THRSH[$SIZE]}
    DATA_SPEC="${INSTANCE_TYPE}_${COST_STRUCTURE}_${SIZE}_${SIZE}_B${MAX_QUANT}_Theta${THETA}_BF${BALANCE_FACTOR}"
    TRAIN_DATA_SPEC="${INSTANCE_TYPE}_${COST_STRUCTURE}_${TRAIN_SIZE}_${TRAIN_SIZE}_B${MAX_QUANT}_Theta${THETA}_BF${BALANCE_FACTOR}"
    MODEL_PATH="$MODEL_DIR/$TRAIN_DATA_SPEC/$MODEL_SPEC"
    for THRSH_VAL in ${THRSH_VALS[@]}
    do
        echo $THRSH_VAL
        # Adjust decoder and decoder parameters to evaluate pipeline with different solver
        python $CODE_DIR/scripts/03_benchmarking/01_run_benchmarking_experiments.py ++instance_dir=$DATA_DIR/$DATA_SPEC ++solution_dir=$SOLUTION_DIR/$DATA_SPEC method="ml-reduction" method.model_path=$MODEL_PATH method.model_name=$MODEL_NAME method.threshold_type="size" method.size_threshold=[$THRSH_VAL] decoder="exact" num_threads=$NUM_CORES decoder.grb_timeout=${TIME_LIMIT[$SIZE]}
    done
done
