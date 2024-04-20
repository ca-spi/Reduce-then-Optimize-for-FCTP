""" Benchmark FCTP algorithms. """

from datetime import datetime
import gzip
import logging
import os
import pickle as pkl
import random
from time import time

import hydra
import numpy as np
from omegaconf import DictConfig
from omegaconf import OmegaConf
import torch

from core.utils.fctp import CapacitatedFCTP
from core.utils.fctp import FixedStepFCTP
from core.fctp_solvers.ip_grb import fctp
from core.fctp_solvers.ip_grb import capacitated_fctp
from core.fctp_solvers.ip_grb import fixed_step_fctp
from core.fctp_solvers.ip_grb import tp
from core.fctp_solvers.ip_grb import get_fctp_bfs
from core.fctp_solvers.ip_grb import sol_vals
from core.fctp_solvers.heuristics import k_shortest_edges_predictor
from core.fctp_solvers.heuristics import random_edges_predictor
from core.fctp_solvers.heuristics import add_feasible_sol_connections
from core.fctp_heuristics_julia.python_wrapper import TabuSearchJuliaEnv
from core.fctp_heuristics_julia.python_wrapper import EvolutionaryAlgorithmJuliaEnv
from core.ml_models.wrapper import ml_based_fctp_reduction
from core.ml_models.wrapper import solve_reduced_problem
from core.utils.ml_utils import load_edge_predictor_model
from core.data_processing.data_utils import load_instance
from core.utils.utils import flatten_dict
from core.evaluation.benchmarking_utils import get_performance_table


def get_k_vals(size_thresholds, m, n):
    """Translate relative size thresholds into absolute size thresholds.

    Parameters
    ----------
    size_thresholds: list
        List of relative size threshold.
    m: int
        Number of supply nodes.
    n: int
        Number of demand nodes.

    Returns
    -------
    k_vals: list
        List of absolute size thresholds.

    """
    num_edges = m * n
    k_vals = [round(num_edges * p) for p in size_thresholds]
    return k_vals


def save_results(path, result_dict):
    """Helper function to save benchmarking results.

    Parameters
    ----------
    path: str
        Path to save results.
    result_dict: dict
        Dictionary with benchmarking information and results.

    Returns
    -------
    None

    """
    with gzip.open(path, "wb") as file:
        pkl.dump(
            result_dict,
            file,
        )


@hydra.main(
    version_base=None,
    config_path=os.path.join(os.getcwd(), "configs", "benchmarking"),
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    """Run benchmarking experiments."""

    # Set all random seeds
    seed = cfg.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Set up solution directory
    os.makedirs(cfg.solution_dir, exist_ok=True)

    # Set up method
    method = cfg.method.name

    # Set up logger
    os.makedirs(cfg.log_dir, exist_ok=True)
    logger = logging.getLogger()
    logger.addHandler(
        logging.FileHandler(
            os.path.join(
                cfg.log_dir,
                f"{datetime.now().strftime('%Y%m%d_%H:%M:%S')}_benchmarking.log",
            ),
            mode="w",
        )
    )

    logger.info(f"Experiment parameters: {cfg}")

    # Julia setup
    os.environ["JULIA_NUM_THREADS"] = str(cfg.num_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(cfg.num_threads)
    if method == "ts":
        logger.info("Initializing Julia environment for TS...")
        ts_env = TabuSearchJuliaEnv()
    elif method == "ea":
        logger.info("Initializing Julia environment for EA...")
        ea_env = EvolutionaryAlgorithmJuliaEnv()

    # Load prediction model
    if cfg.method.get("model_path") is not None:
        edge_predictor_model = (
            load_edge_predictor_model(cfg.method.model_path, get_feature_fun=True),
            cfg.method.model_name,
        )

    # Set up decoder for problem reduction approaches
    if cfg.get("decoder") is not None:
        decoder = cfg.decoder.name
        decoder_cfg = OmegaConf.to_container(cfg.decoder)
        decoder_env = None
        del decoder_cfg["name"]
        decoder_param_spec = "_".join(
            [f"{k}_{v}" for k, v in flatten_dict(decoder_cfg).items()]
        )
        if decoder == "ts":
            decoder_cfg["tabu_in_range"] = tuple(decoder_cfg["tabu_in_range"].values())
            decoder_cfg["tabu_out_range"] = tuple(
                decoder_cfg["tabu_out_range"].values()
            )
            decoder_cfg["seed"] = seed
            logger.info("Initializing Julia environment for TS...")
            decoder_env = TabuSearchJuliaEnv()
        elif decoder == "ea":
            decoder_cfg["seed"] = seed
            logger.info("Initializing Julia environment for EA...")
            decoder_env = EvolutionaryAlgorithmJuliaEnv()
        elif decoder in ["exact", "lp"]:
            decoder_cfg["grb_threads"] = cfg.num_threads
            decoder_cfg["grb_verbosity"] = cfg.verbose

    # Process all instances
    instance_paths = [
        os.path.join(cfg.instance_dir, filename)
        for filename in os.listdir(cfg.instance_dir)
    ]
    logger.info(f"{len(instance_paths)} benchmark instances")

    for counter, instance_path in enumerate(instance_paths):

        instance_id = instance_path.split("/")[-1].split(".")[0].split("_")[-1]
        solution_filename = f"sol_instance_{instance_id}.pkl.gz"

        # Initialize result dict
        result_dict = {
            "instance_path": instance_path,
            "method": method,
            "experiment_config": cfg,
        }

        # Print progress
        logger.info(
            f"Processing instance {instance_id} ({counter+1}/{len(instance_paths)})..."
        )

        # Load instance
        instance = load_instance(instance_path)

        # Solve exactly (Gurobi)
        if method == "exact":
            grb_cfg = OmegaConf.to_container(cfg.method)
            del grb_cfg["name"]
            param_spec = "_".join([f"{k}_{v}" for k, v in grb_cfg.items()])
            start = time()
            if isinstance(instance, CapacitatedFCTP):
                model, x, _ = capacitated_fctp(
                    instance.supply,
                    instance.demand,
                    instance.var_costs,
                    instance.fix_costs,
                    instance.edge_capacities,
                )
            elif isinstance(instance, FixedStepFCTP):
                model, x, _ = fixed_step_fctp(
                    instance.supply,
                    instance.demand,
                    instance.var_costs,
                    instance.fix_costs,
                    instance.vehicle_capacities,
                )
            else:
                model, x, _ = fctp(
                    instance.supply,
                    instance.demand,
                    instance.var_costs,
                    instance.fix_costs,
                )
            model.setParam("OutputFlag", 0)
            model.setParam("TimeLimit", cfg.method.grb_timeout)
            if cfg.num_threads is not None:
                model.setParam("Threads", cfg.num_threads)
            model.setParam("Seed", seed)
            model.optimize()
            sol = sol_vals(x)
            runtime = time() - start
            assert np.sum(list(sol.values())) == np.sum(instance.demand)
            result_dict.update(
                {
                    "solution": sol,
                    "objective_value": instance.eval_sol_dict(sol),
                    "runtime": runtime,
                    "solver_runtime": model.Runtime,
                    "solver_status": model.Status,
                    "mip_gap": model.MIPGap,
                }
            )
            solution_path = os.path.join(cfg.solution_dir, method, param_spec)
            os.makedirs(solution_path, exist_ok=True)
            save_results(os.path.join(solution_path, solution_filename), result_dict)

        # Solve with EA (Eckert & Gottlieb, 2002)
        elif method == "ea":
            if isinstance(instance, CapacitatedFCTP) or isinstance(
                instance, FixedStepFCTP
            ):
                raise NotImplementedError
            ea_cfg = OmegaConf.to_container(cfg.method)
            del ea_cfg["name"]
            param_spec = "_".join([f"{k}_{v}" for k, v in ea_cfg.items()])
            ea_cfg["seed"] = seed
            sol, sol_val, runtime = ea_env.run(instance, ea_cfg)
            assert sol_val == instance.eval_sol_dict(
                sol
            ), "Inconsistent objective function values"
            assert np.sum(list(sol.values())) == np.sum(instance.demand)
            result_dict.update(
                {
                    "solution": sol,
                    "objective_value": sol_val,
                    "runtime": runtime,
                }
            )
            solution_path = os.path.join(cfg.solution_dir, method, param_spec)
            os.makedirs(solution_path, exist_ok=True)
            save_results(os.path.join(solution_path, solution_filename), result_dict)

        # Solve with TS (Sun et al., 1998)
        elif method == "ts":
            if isinstance(instance, CapacitatedFCTP) or isinstance(
                instance, FixedStepFCTP
            ):
                raise NotImplementedError
            ts_cfg = OmegaConf.to_container(cfg.method)
            del ts_cfg["name"]
            param_spec = "_".join([f"{k}_{v}" for k, v in flatten_dict(ts_cfg).items()])
            ts_cfg["tabu_in_range"] = tuple(ts_cfg["tabu_in_range"].values())
            ts_cfg["tabu_out_range"] = tuple(ts_cfg["tabu_out_range"].values())
            ts_cfg["seed"] = seed
            bfs = get_fctp_bfs(instance)
            sol, sol_val, runtime = ts_env.run(instance, bfs, ts_cfg)
            assert sol_val == instance.eval_sol_dict(
                sol
            ), "Inconsistent objective function values"
            assert np.sum(list(sol.values())) == np.sum(instance.demand)
            result_dict.update(
                {
                    "solution": sol,
                    "objective_value": sol_val,
                    "runtime": runtime,
                }
            )
            solution_path = os.path.join(cfg.solution_dir, method, param_spec)
            os.makedirs(solution_path, exist_ok=True)
            save_results(os.path.join(solution_path, solution_filename), result_dict)

        # Solve linear relaxation
        elif method == "linear-relax":
            start = time()
            if isinstance(instance, CapacitatedFCTP):
                model, x, _ = capacitated_fctp(
                    instance.supply,
                    instance.demand,
                    instance.var_costs,
                    instance.fix_costs,
                    instance.edge_capacities,
                    relax=True,
                )
            elif isinstance(instance, FixedStepFCTP):
                model, x, _ = fixed_step_fctp(
                    instance.supply,
                    instance.demand,
                    instance.var_costs,
                    instance.fix_costs,
                    instance.vehicle_capacities,
                    relax=True,
                )
            else:
                model, x, _ = fctp(
                    instance.supply,
                    instance.demand,
                    instance.var_costs,
                    instance.fix_costs,
                    relax=True,
                )
            model.setParam("OutputFlag", 0)
            model.setParam("TimeLimit", cfg.method.grb_timeout)
            if cfg.num_threads is not None:
                model.setParam("Threads", cfg.num_threads)
            model.setParam("Seed", seed)
            model.optimize()
            sol = sol_vals(x)
            runtime = time() - start
            assert np.sum(list(sol.values())) == np.sum(instance.demand)
            result_dict.update(
                {
                    "solution": sol,
                    "objective_value": instance.eval_sol_dict(sol),
                    "runtime": runtime,
                    "solver_runtime": model.Runtime,
                    "solver_status": model.Status,
                }
            )
            solution_path = os.path.join(cfg.solution_dir, method)
            os.makedirs(solution_path, exist_ok=True)
            save_results(os.path.join(solution_path, solution_filename), result_dict)

        # Solve with linear TP (only variable costs)
        elif method == "linear-tp":
            if isinstance(instance, CapacitatedFCTP) or isinstance(
                instance, FixedStepFCTP
            ):
                raise NotImplementedError
            start = time()
            model, x = tp(instance.supply, instance.demand, costs=instance.var_costs)
            model.setParam("OutputFlag", 0)
            model.setParam("TimeLimit", cfg.method.grb_timeout)
            if cfg.num_threads is not None:
                model.setParam("Threads", cfg.num_threads)
            model.setParam("Seed", seed)
            model.optimize()
            sol = sol_vals(x)
            runtime = time() - start
            assert np.sum(list(sol.values())) == np.sum(instance.demand)
            result_dict.update(
                {
                    "solution": sol,
                    "objective_value": instance.eval_sol_dict(sol),
                    "runtime": runtime,
                    "solver_runtime": model.Runtime,
                    "solver_status": model.Status,
                }
            )
            solution_path = os.path.join(cfg.solution_dir, method)
            os.makedirs(solution_path, exist_ok=True)
            save_results(os.path.join(solution_path, solution_filename), result_dict)

        # Solve with k-random-connections selection
        elif method == "k-random-edges":
            if isinstance(instance, CapacitatedFCTP) or isinstance(
                instance, FixedStepFCTP
            ):
                raise NotImplementedError
            k_vals = get_k_vals(cfg.method.size_threshold, instance.m, instance.n)
            for i, k_val in enumerate(k_vals):
                thrsh = cfg.method.size_threshold[i]
                start = time()
                relevant_connections = random_edges_predictor(
                    (instance.m, instance.n),
                    k=k_val,
                )
                num_edges_pred = np.sum(relevant_connections)
                relevant_connections = add_feasible_sol_connections(
                    relevant_connections,
                    instance,
                    method="nwc",
                )
                num_edges_enriched = np.sum(relevant_connections)
                sol, solver_runtime, solver_status, mip_gap = solve_reduced_problem(
                    instance,
                    relevant_connections,
                    decoder,
                    decoder_cfg,
                    decoder_env,
                    seed,
                )
                runtime = time() - start
                assert np.sum(list(sol.values())) == np.sum(instance.demand)
                result_dict_k = result_dict.copy()
                result_dict_k.update(
                    {
                        "solution": sol,
                        "objective_value": instance.eval_sol_dict(sol),
                        "runtime": runtime,
                        "solver_runtime": solver_runtime,
                        "num_edges_pred": num_edges_pred,
                        "num_edges_enriched": num_edges_enriched,
                        "method_param": k_val,
                    }
                )
                if solver_status is not None:
                    result_dict_k["solver_status"] = solver_status
                if mip_gap is not None:
                    result_dict_k["mip_gap"] = mip_gap
                solution_path = os.path.join(
                    cfg.solution_dir,
                    method,
                    f"{decoder}-{decoder_param_spec}",
                    str(thrsh),
                )
                os.makedirs(solution_path, exist_ok=True)
                save_results(
                    os.path.join(solution_path, solution_filename), result_dict_k
                )

        # Solve with k-shortest-connections selection
        elif method == "k-shortest-edges":
            if isinstance(instance, CapacitatedFCTP) or isinstance(
                instance, FixedStepFCTP
            ):
                raise NotImplementedError
            k_vals = get_k_vals(cfg.method.size_threshold, instance.m, instance.n)
            for i, k_val in enumerate(k_vals):
                thrsh = cfg.method.size_threshold[i]
                start = time()
                relevant_connections = k_shortest_edges_predictor(
                    instance,
                    k=k_val,
                )
                num_edges_pred = np.sum(relevant_connections)
                relevant_connections = add_feasible_sol_connections(
                    relevant_connections,
                    instance,
                    method="lcm",
                )
                num_edges_enriched = np.sum(relevant_connections)
                sol, solver_runtime, solver_status, mip_gap = solve_reduced_problem(
                    instance,
                    relevant_connections,
                    decoder,
                    decoder_cfg,
                    decoder_env,
                    seed,
                )
                runtime = time() - start
                assert np.sum(list(sol.values())) == np.sum(instance.demand)
                result_dict_k = result_dict.copy()
                result_dict_k.update(
                    {
                        "solution": sol,
                        "objective_value": instance.eval_sol_dict(sol),
                        "runtime": runtime,
                        "solver_runtime": solver_runtime,
                        "num_edges_pred": num_edges_pred,
                        "num_edges_enriched": num_edges_enriched,
                        "method_param": k_val,
                    }
                )
                if solver_status is not None:
                    result_dict_k["solver_status"] = solver_status
                if mip_gap is not None:
                    result_dict_k["mip_gap"] = mip_gap
                solution_path = os.path.join(
                    cfg.solution_dir,
                    method,
                    f"{decoder}-{decoder_param_spec}",
                    str(thrsh),
                )
                os.makedirs(solution_path, exist_ok=True)
                save_results(
                    os.path.join(solution_path, solution_filename), result_dict_k
                )

        elif method == "ml-reduction":
            # Threshold type and values
            threshold_type = cfg.method.threshold_type
            if threshold_type == "size":
                thresholds = cfg.method.size_threshold
            elif threshold_type == "prob":
                thresholds = cfg.method.prob_threshold
            else:
                raise ValueError

            for thrsh in thresholds:
                start = time()
                (
                    sol,
                    num_edges_pred,
                    num_edges_enriched,
                    solver_runtime,
                    solver_status,
                    mip_gap,
                ) = ml_based_fctp_reduction(
                    instance,
                    predictor_model=edge_predictor_model[0],
                    threshold_type=threshold_type,
                    threshold=thrsh,
                    decoder=decoder,
                    decoder_cfg=decoder_cfg,
                    decoder_env=decoder_env,
                    seed=seed,
                )
                runtime = time() - start
                assert np.sum(list(sol.values())) == np.sum(instance.demand)
                result_dict_k = result_dict.copy()
                result_dict_k.update(
                    {
                        "solution": sol,
                        "objective_value": instance.eval_sol_dict(sol),
                        "runtime": runtime,
                        "solver_runtime": solver_runtime,
                        "num_edges_pred": num_edges_pred,
                        "num_edges_enriched": num_edges_enriched,
                        "method_param": thrsh,
                        "model": edge_predictor_model[1],
                    }
                )
                if solver_status is not None:
                    result_dict_k["solver_status"] = solver_status
                if mip_gap is not None:
                    result_dict_k["mip_gap"] = mip_gap
                solution_path = os.path.join(
                    cfg.solution_dir,
                    method,
                    threshold_type,
                    f"{decoder}-{decoder_param_spec}",
                    edge_predictor_model[1],
                    str(thrsh),
                )
                os.makedirs(solution_path, exist_ok=True)
                save_results(
                    os.path.join(solution_path, solution_filename), result_dict_k
                )

    logger.info("**************** Finished benchmarking ****************")

    # logger.info summary table
    if cfg.summarize:
        logger.info("Preparing summary table...")
        df = get_performance_table(cfg.solution_dir)
        os.makedirs(cfg.result_dir, exist_ok=True)
        df.to_csv(os.path.join(cfg.result_dir, "benchmarking_summary.csv"))


if __name__ == "__main__":
    main()
