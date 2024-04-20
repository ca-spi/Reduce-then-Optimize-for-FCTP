#!/bin/bash

CODE_DIR=$PWD
DATA_DIR="$CODE_DIR/data/instances/benchmarking"
SOLUTION_DIR="$CODE_DIR/benchmarking"

SEED=0
NUM_CORES=1
TIME_LIMIT=43200

# Dataset configuration (adjust to evaluate different dataset)
INSTANCE_TYPE="fctp"
COST_STRUCTURE="agarwal-aneja"
SIZE=15
MAX_QUANT=20
THETA=0.2
BALANCE_FACTOR=1.0
DATA_SPEC="${INSTANCE_TYPE}_${COST_STRUCTURE}_${SIZE}_${SIZE}_B${MAX_QUANT}_Theta${THETA}_BF${BALANCE_FACTOR}"

for METHOD in "k-random-edges" "k-shortest-edges"
do
    for THRSH in 0.2 0.25 0.3 0.35 0.4 0.45 0.5
    do
        # Adjust decoder and decoder parameters to evaluate pipeline with different solver
        python $CODE_DIR/scripts/03_benchmarking/01_run_benchmarking_experiments.py ++instance_dir=$DATA_DIR/$DATA_SPEC ++solution_dir=$SOLUTION_DIR/$DATA_SPEC method=$METHOD method.size_threshold=[$THRSH] decoder="exact" num_threads=$NUM_CORES decoder.grb_timeout=$TIME_LIMIT
    done
done
