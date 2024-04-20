#!/bin/bash

CODE_DIR=$PWD
DATA_DIR="$CODE_DIR/data/instances/benchmarking"
SOLUTION_DIR="$CODE_DIR/benchmarking"
MODEL_DIR="$CODE_DIR/trained_models/sol_edge_predictor"

SEED=0

TIME_LIMIT=43200
NUM_CORES=1

# Benchmarking dataset configurations
B50=("fctp" "agarwal-aneja" 15 50 0.2 1.0)
UB10=("fctp" "agarwal-aneja" 15 20 0.2 1.1)
UB30=("fctp" "agarwal-aneja" 15 20 0.2 1.3)
RBM15=("fctp" "roberti" 15 20 0.2 1.0)
EUCL=("fctp" "euclidean" 15 20 0.2 1.0)
VFCR05=("fctp" "agarwal-aneja" 15 20 0.5 1.0)
VFCR00=("fctp" "agarwal-aneja" 15 20 0.0 1.0)
CFCTP=("c-fctp" "agarwal-aneja" 15 20 0.2 1.0)
FSFCTP=("fs-fctp" "agarwal-aneja" 15 20 0.2 1.0)
declare -A DATASETS
DATASETS[B50]=B50[@]
DATASETS[UB10]=UB10[@]
DATASETS[UB30]=UB30[@]
DATASETS[RBM15]=RBM15[@]
DATASETS[EUCL]=EUCL[@]
DATASETS[VFCR05]=VFCR05[@]
DATASETS[VFCR00]=VFCR00[@]
DATASETS[CFCTP]=CFCTP[@]
DATASETS[FSFCTP]=FSFCTP[@]

# Model path and name (adjust to evaluate different model)
MODEL_SPEC="gcnn/model_gcnn_features_bipartite_raw_prediction_task_binary_classification_normalization_standard_hidden_layer_dim_20_num_conv_layers_10_num_dense_layers_2/cross_val/best_checkpoint.pth.tar"
MODEL_NAME="model_gcnn_features_bipartite_raw_prediction_task_binary_classification_normalization_standard_hidden_layer_dim_20_num_conv_layers_10_num_dense_layers_2"

for i in "${!DATASETS[@]}"
do
    IFS=' ' read -r -a CONFIG <<< ${!DATASETS[$i]}
    INSTANCE_TYPE=${CONFIG[0]}
    COST_STRUCTURE=${CONFIG[1]}
    SIZE=${CONFIG[2]}
    MAX_QUANT=${CONFIG[3]}
    THETA=${CONFIG[4]}
    BALANCE_FACTOR=${CONFIG[5]}
    DATA_SPEC="${INSTANCE_TYPE}_${COST_STRUCTURE}_${SIZE}_${SIZE}_B${MAX_QUANT}_Theta${THETA}_BF${BALANCE_FACTOR}"
    MODEL_PATH="$MODEL_DIR/$DATA_SPEC/$MODEL_SPEC"
    # Solve full problem (GRB)
    python $CODE_DIR/scripts/03_benchmarking/01_run_benchmarking_experiments.py ++instance_dir=$DATA_DIR/$DATA_SPEC ++solution_dir=$SOLUTION_DIR/$DATA_SPEC method="exact" num_threads=$NUM_CORES method.grb_timeout=$TIME_LIMIT
    # Reduce-then-Optimize
    for THRSH in 0.2 0.3 0.4
    do
        python $CODE_DIR/scripts/03_benchmarking/01_run_benchmarking_experiments.py ++instance_dir=$DATA_DIR/$DATA_SPEC ++solution_dir=$SOLUTION_DIR/$DATA_SPEC method="ml-reduction" method.model_path=$MODEL_PATH method.model_name=$MODEL_NAME method.threshold_type="size" method.size_threshold=[$THRSH] decoder="exact" num_threads=$NUM_CORES decoder.grb_timeout=$TIME_LIMIT
    done
done