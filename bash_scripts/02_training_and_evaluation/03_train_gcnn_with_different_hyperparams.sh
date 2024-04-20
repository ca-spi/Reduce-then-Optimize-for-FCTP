#!/bin/bash

CODE_DIR=$PWD
DATA_DIR="$CODE_DIR/data/samples"
MODEL_DIR="$CODE_DIR/trained_models/sol_edge_predictor"

SEED=0
NUM_CORES=4

MODEL_NAME="gcnn"

# Training dataset configuration
INSTANCE_TYPE="fctp"
COST_STRUCTURE="agarwal-aneja"
SIZE=15
MAX_QUANT=20
THETA=0.2
BALANCE_FACTOR=1.0
DATA_SPEC="${INSTANCE_TYPE}_${COST_STRUCTURE}_${SIZE}_${SIZE}_B${MAX_QUANT}_Theta${THETA}_BF${BALANCE_FACTOR}"

# Hyperparameter configurations
CONV1=(1 2 20)
CONV2=(3 2 20)
CONV3=(5 2 20)
CONV4=(10 2 20)
CONV5=(20 2 20)
DENSE1=(10 1 20)
DENSE2=(10 5 20)
DIM1=(10 2 1)
DIM2=(10 2 3)
DIM3=(10 2 5)
DIM4=(10 2 10)
DIM5=(10 2 50)
declare -A CONFIGS
CONFIGS[CONV1]=CONV1[@]
CONFIGS[CONV2]=CONV2[@]
CONFIGS[CONV3]=CONV3[@]
CONFIGS[CONV4]=CONV4[@]
CONFIGS[CONV5]=CONV5[@]
CONFIGS[DENSE1]=DENSE1[@]
CONFIGS[DENSE2]=DENSE2[@]
CONFIGS[DIM1]=DIM1[@]
CONFIGS[DIM2]=DIM2[@]
CONFIGS[DIM3]=DIM3[@]
CONFIGS[DIM4]=DIM4[@]
CONFIGS[DIM5]=DIM5[@]

for i in "${!CONFIGS[@]}"
do
    IFS=' ' read -r -a CONFIG <<< ${!CONFIGS[$i]}
    NUM_CONV=${CONFIG[0]}
    NUM_DENSE=${CONFIG[1]}
    DIM=${CONFIG[2]}   
    python $CODE_DIR/scripts/02_training_and_evaluation/01_train_sol_edge_predictor.py data_path=$DATA_DIR/$DATA_SPEC model=$MODEL_NAME model.num_conv_layers=$NUM_CONV model.num_dense_layers=$NUM_DENSE model.hidden_layer_dim=$DIM out_dir=$MODEL_DIR/$DATA_SPEC/$MODEL_NAME seed=$SEED cross_validate=false max_num_epochs=200 early_stopping=200 lr_decay=false
done

