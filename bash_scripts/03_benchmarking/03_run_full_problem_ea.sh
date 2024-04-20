#!/bin/bash

CODE_DIR=$PWD
DATA_DIR="$CODE_DIR/data/instances/benchmarking"
SOLUTION_DIR="$CODE_DIR/benchmarking"

SEED=0

NUM_CORES=1

# Dataset configuration (BASE)
INSTANCE_TYPE="fctp"
COST_STRUCTURE="agarwal-aneja"
MAX_QUANT=20
THETA=0.2
BALANCE_FACTOR=1.0

# EA parameters
declare -A MAX_SOLS
MAX_SOLS[15]=100000
MAX_SOLS[30]=1000000
MAX_SOLS[120]=1000000
declare -A PATIENCE
PATIENCE[15]=1000000
PATIENCE[30]=1000000
PATIENCE[120]=1000000
declare -A POP_SIZE
POP_SIZE[15]=100
POP_SIZE[30]=200
POP_SIZE[120]=200

for SIZE in 15 30 120
do
    DATA_SPEC="${INSTANCE_TYPE}_${COST_STRUCTURE}_${SIZE}_${SIZE}_B${MAX_QUANT}_Theta${THETA}_BF${BALANCE_FACTOR}"
    python $CODE_DIR/scripts/03_benchmarking/01_run_benchmarking_experiments.py ++instance_dir=$DATA_DIR/$DATA_SPEC ++solution_dir=$SOLUTION_DIR/$DATA_SPEC method="ea" method.max_unique_sols=${MAX_SOLS[${SIZE}]} method.patience=${PATIENCE[${SIZE}]} method.pop_size=${POP_SIZE[${SIZE}]} num_threads=$NUM_CORES
done