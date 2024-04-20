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

# TS parameters
L=5
declare -A BETA
BETA[15]=0.5
BETA[30]=0.5
BETA[120]=0.1
declare -A GAMMA
GAMMA[15]=0.5
GAMMA[30]=0.5
GAMMA[120]=0.1
declare -A TABU_IN_LB
TABU_IN_LB[15]=7
TABU_IN_LB[30]=7
TABU_IN_LB[120]=7
declare -A TABU_IN_UB
TABU_IN_UB[15]=10
TABU_IN_UB[30]=10
TABU_IN_UB[120]=10
declare -A TABU_OUT_LB
TABU_OUT_LB[15]=2
TABU_OUT_LB[30]=2
TABU_OUT_LB[120]=5
declare -A TABU_OUT_UB
TABU_OUT_UB[15]=4
TABU_OUT_UB[30]=4
TABU_OUT_UB[120]=7

for SIZE in 15 30 120
do
    DATA_SPEC="${INSTANCE_TYPE}_${COST_STRUCTURE}_${SIZE}_${SIZE}_B${MAX_QUANT}_Theta${THETA}_BF${BALANCE_FACTOR}"
    python $CODE_DIR/scripts/03_benchmarking/01_run_benchmarking_experiments.py ++instance_dir=$DATA_DIR/$DATA_SPEC ++solution_dir=$SOLUTION_DIR/$DATA_SPEC method="ts" method.L=$L method.tabu_in_range.lb=${TABU_IN_LB[${SIZE}]} method.tabu_in_range.ub=${TABU_IN_UB[${SIZE}]} method.tabu_out_range.lb=${TABU_OUT_LB[${SIZE}]} method.tabu_out_range.ub=${TABU_OUT_UB[${SIZE}]} method.beta=${BETA[${SIZE}]} method.gamma=${GAMMA[${SIZE}]} num_threads=$NUM_CORES
done