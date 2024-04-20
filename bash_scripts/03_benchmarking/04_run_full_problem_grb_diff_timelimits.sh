#!/bin/bash

CODE_DIR=$PWD
DATA_DIR="$CODE_DIR/data/instances/benchmarking"
SOLUTION_DIR="$CODE_DIR/benchmarking"

SEED=0
NUM_CORES=8

# Runtime limits
declare -A TIME_LIMIT
TIME_LIMIT[30]="5 10 30 60"
TIME_LIMIT[120]="10 30 60 300"

# Dataset configuration (BASE)
INSTANCE_TYPE="fctp"
COST_STRUCTURE="agarwal-aneja"
MAX_QUANT=20
THETA=0.2
BALANCE_FACTOR=1.0

for SIZE in 30 120
do
    IFS=' ' read -r -a TL_LIST <<< ${TIME_LIMIT[$SIZE]}
    for TL in ${TL_LIST[@]}
    do
        DATA_SPEC="${INSTANCE_TYPE}_${COST_STRUCTURE}_${SIZE}_${SIZE}_B${MAX_QUANT}_Theta${THETA}_BF${BALANCE_FACTOR}"
        python $CODE_DIR/scripts/03_benchmarking/01_run_benchmarking_experiments.py ++instance_dir=$DATA_DIR/$DATA_SPEC ++solution_dir=$SOLUTION_DIR/$DATA_SPEC method="exact" num_threads=$NUM_CORES method.grb_timeout=$TL
    done
done

