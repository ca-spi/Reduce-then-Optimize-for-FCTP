#!/bin/bash

CODE_DIR=$PWD
DATA_DIR="$CODE_DIR/data/instances/benchmarking"
SOLUTION_DIR="$CODE_DIR/benchmarking"

SEED=0

TIME_LIMIT=43200
declare -A NUM_CORES
NUM_CORES[15]=1
NUM_CORES[30]=8
NUM_CORES[120]=8

# Dataset configuration (BASE)
INSTANCE_TYPE="fctp"
COST_STRUCTURE="agarwal-aneja"
MAX_QUANT=20
THETA=0.2
BALANCE_FACTOR=1.0

for SIZE in 15 30 120
do
    DATA_SPEC="${INSTANCE_TYPE}_${COST_STRUCTURE}_${SIZE}_${SIZE}_B${MAX_QUANT}_Theta${THETA}_BF${BALANCE_FACTOR}"
    for METHOD in "exact" "linear-relax" "linear-tp"
    do
        python $CODE_DIR/scripts/03_benchmarking/01_run_benchmarking_experiments.py ++instance_dir=$DATA_DIR/$DATA_SPEC ++solution_dir=$SOLUTION_DIR/$DATA_SPEC method=$METHOD num_threads=${NUM_CORES[$SIZE]} method.grb_timeout=$TIME_LIMIT
    done
done

