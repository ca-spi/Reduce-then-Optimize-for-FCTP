#!/bin/bash

CODE_DIR=$PWD
DATA_DIR="$CODE_DIR/data"

NUM_CORES=2
SEED=0
NUM_INSTANCES=4000

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
    SPEC="${INSTANCE_TYPE}_${COST_STRUCTURE}_${SIZE}_${SIZE}_B${MAX_QUANT}_Theta${THETA}_BF${BALANCE_FACTOR}"
    # Generate instances
    python $CODE_DIR/scripts/01_data/01_generate_instances.py --num_instances $NUM_INSTANCES --instance_type $INSTANCE_TYPE --cost_structure $COST_STRUCTURE --size $SIZE --max_quantity $MAX_QUANT --theta $THETA --supply_demand_ratio $BALANCE_FACTOR --dir "${DATA_DIR}/instances/training/${SPEC}" --seed $SEED
    # Generate samples
    python $CODE_DIR/scripts/01_data/02_generate_samples.py --instance_dir "${DATA_DIR}/instances/training/${SPEC}" --sample_dir "${DATA_DIR}/samples/${SPEC}" --grb_timeout 600 --seed $SEED --num_threads $NUM_CORES
done