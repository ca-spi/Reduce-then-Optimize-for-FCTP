#!/bin/bash

CODE_DIR=$PWD
DATA_DIR="$CODE_DIR/data/samples"
MODEL_DIR="$CODE_DIR/trained_models/sol_edge_predictor"

CROSS_VAL=true
SEED=0
NUM_CORES=4
MODEL_NAME="gcnn"

# Training dataset configurations
BASE=("fctp" "agarwal-aneja" 15 20 0.2 1.0)
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
DATASETS[BASE]=BASE[@]
DATASETS[B50]=B50[@]
DATASETS[UB10]=UB10[@]
DATASETS[UB30]=UB30[@]
DATASETS[RBM15]=RBM15[@]
DATASETS[EUCL]=EUCL[@]
DATASETS[VFCR05]=VFCR05[@]
DATASETS[VFCR00]=VFCR00[@]
DATASETS[CFCTP]=CFCTP[@]
DATASETS[FSFCTP]=FSFCTP[@]

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
    python $CODE_DIR/scripts/02_training_and_evaluation/01_train_sol_edge_predictor.py data_path=$DATA_DIR/$DATA_SPEC model=$MODEL_NAME out_dir=$MODEL_DIR/$DATA_SPEC/$MODEL_NAME seed=$SEED cross_validate=$CROSS_VAL
done