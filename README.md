# Reduce-then-Optimize for the Fixed-Charge Transportation Problem (FCTP)

This repository contains the code to reproduce the results presented in the paper "Reduce-then-Optimize for the Fixed-Charge Transportation Problem" by Spieckermann, Minner, and Schiffer (2024).

## Docker Setup

We recommend using Docker to replicate our developing and testing environment:
```bash 
# Build
docker build -t fctp-reduce-then-optimize:latest .

# Test
docker run -it --rm -v $PWD:/code -w /code fctp-reduce-then-optimize:latest bash
```

Note: If Gurobi is run with a commercial or academic license, mount the license file, e.g., `-v $HOME/gurobi.lic:/opt/gurobi/gurobi.lic:ro`.

To execute scripts, run from root directory:
```bash
PYTHONPATH=$PWD python scripts/<example_script.py>
```

Using docker:
```bash
docker run --rm -v $PWD:/code -w /code -e PYTHONPATH=/code fctp-reduce-then-optimize:latest python scripts/<example_script.py>
```

## Step 1: Data Generation

Example:
```bash
# Generate instances (for training or benchmarking)
docker run --rm -v $PWD:/code -w /code -e PYTHONPATH=/code fctp-reduce-then-optimize:latest python scripts/01_data/01_generate_instances.py --num_instances 30 --cost_structure "agarwal-aneja" --size 15 --max_quantity 20 --theta 0.2 --supply_demand_ratio 1.0 --dir "data/instances" --seed 0

# Generate training samples by solving instances to optimality
docker run --rm -v $PWD:/code -w /code -e PYTHONPATH=/code fctp-reduce-then-optimize:latest python scripts/01_data/02_generate_samples.py --instance_dir "data/instances" --sample_dir "data/samples" --grb_timeout 600 --seed 0
```

Use bash scripts `bash_scripts/01_data/01_generate_training_samples.sh` and `bash_scripts/01_data/02_generate_benchmarking_instances.sh` to run all data set configurations used in paper.
```bash
# Generate training instances and samples
docker run --rm -v $PWD:/code -w /code -e PYTHONPATH=/code fctp-reduce-then-optimize:latest ./bash_scripts/01_data/01_generate_training_samples.sh

# Generate benchmarking instances
docker run --rm -v $PWD:/code -w /code -e PYTHONPATH=/code fctp-reduce-then-optimize:latest ./bash_scripts/01_data/02_generate_benchmarking_instances.sh
```

---
**NOTE**
Make sure to use different seeds when generating training and benchmarking instances to prevent leakage. Consider running different datasets and seeds in parallel to speed up sample generation. The original data of the paper is provided in `data_paper`.

---

## Step 2: Training and Model Selection

Example:
```bash
docker run --rm -v $PWD:/code -w /code -e PYTHONPATH=/code fctp-reduce-then-optimize:latest python scripts/02_training_and_evaluation/01_train_sol_edge_predictor.py data_path="data/samples" model="gcnn" model.num_conv_layers=10 model.num_dense_layers=2 model.hidden_layer_dim=20 out_dir="trained_models" seed=0 cross_validate=true
```

Use `bash_scripts/02_training_and_evaluation/01_train_gcnn_sol_edge_predictor.sh` to train GNN on different data sets. Use `bash_scripts/02_training_and_evaluation/02_train_baselines.sh` to train baseline ML models on BASE instances.

## Step 3: Benchmarking

Example:
```bash
# Example 1: Solve full problem with Gurobi
docker run --rm -v $PWD:/code -w /code -e PYTHONPATH=/code fctp-reduce-then-optimize:latest python scripts/03_benchmarking/01_run_benchmarking_experiments.py ++instance_dir="data/instances/benchmarking" ++solution_dir="benchmarking" method="exact" num_threads=1 method.grb_timeout=60

# Example 2: Solve full problem with TS
docker run --rm -v $PWD:/code -w /code -e PYTHONPATH=/code fctp-reduce-then-optimize:latest python scripts/03_benchmarking/01_run_benchmarking_experiments.py ++instance_dir="data/instances/benchmarking" ++solution_dir="benchmarking" method="ts" method.L=5 num_threads=1

# Example 3: Reduce-then-Optimize with GNN+GRB (assuming a GNN model checkpoint at trained_models/best_checkpoint.pth.tar)
docker run --rm -v $PWD:/code -w /code -e PYTHONPATH=/code fctp-reduce-then-optimize:latest python scripts/03_benchmarking/01_run_benchmarking_experiments.py ++instance_dir="data/instances/benchmarking" ++solution_dir="benchmarking" method="ml-reduction" method.model_path="trained_models/best_checkpoint.pth.tar" method.model_name="my_gnn" method.threshold_type="size" method.size_threshold=[0.2,0.3,0.4] decoder="exact" num_threads=1
```

Use bash scripts under `bash_scripts/03_benchmarking` to reproduce paper experiments. The original benchmarking results of the paper are provided in `benchmarking_paper`.

## Step 4: Analyses and Visualizations

`scripts/04_analyses` contains all scripts needed to reproduce the analyses and plots presented the paper (based on the benchmarking results in `benchmarking_paper`).

Plot optimality gap against runtime for reduce-then-optimize pipeline (GNN+GRB):
```bash
docker run --rm -v $PWD:/code -w /code -e PYTHONPATH=/code fctp-reduce-then-optimize:latest python scripts/04_analyses/runtime_vs_optgap_plots.py
```

Print benchmarking performance table to compare different FCTP types (GNN+GRB):
```bash
docker run --rm -v $PWD:/code -w /code -e PYTHONPATH=/code fctp-reduce-then-optimize:latest python scripts/04_analyses/benchmarking_performance_different_datasets_table.py
```

## Additional Experiments and Analyses

### Hyperparameter Screening
Override default values to train GNN with other hyperparameter values. To better assess the risk for overfitting, train model for a fixed number of epochs without learning rate decay and early stopping. Use `bash_scripts/02_training_and_evaluation/03_train_gcnn_with_different_hyperparams.sh` to reproduce paper experiments. Generate comparative plots with `scripts/04_analyses/gcnn_hyperparameter_plots.py`.

## Add Problem Variants
To extend to new problem variants, make the following adjustments:

**Step 1: Instances and Samples**
* Add class to describe instance in `core/utils/fctp.py`.
* Implement MIP formulation as Gurobi model in `core/fctp_solvers/ip_grb.py`.
* Implement random instance generation function in `core/data_processing/instance_generation.py`.
* Extend instance loader in `core/data_processing/data_utils.py`
* Extend sample generator in `core/data_processing/sample_generation.py`
* Adjust instance and sample generation scripts in `scripts/01_data` and `bash_scripts/01_data`.

**Step 2: Features, ML Model, and Training**
* Adjust features (`core/utils/ml_utils.py`) and (if necessary) ML models (`core/ml_models.py`) to represent new problem variant.
* Extend wrapper and helper functions (`core/utils/ml_utils.py`) to new models and features (e.g., model loading).
* Extend training scripts in `scripts/02_training_and_evaluation` and `bash_scripts/02_training_and_evaluation`.

**Step 3: Benchmarking**
* Adjust benchmarking script `scripts/03_benchmarking/01_run_benchmarking_experiments` to include solvers for new problem variant and prevent usage of unsuitable methods.
* Extend reduce-then-optimize wrapper in `core/ml_models/wrapper.py`.
* Extend benchmarking experiment scripts, e.g., `bash_scripts/scripts/03_benchmarking/08_run_diff_instance_types.sh`.
