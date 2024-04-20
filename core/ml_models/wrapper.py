"""Wrapper and helper functions for ML-based FCTP algorithms. """

import time

import numpy as np
import torch

from core.utils.fctp import CapacitatedFCTP
from core.utils.fctp import FixedStepFCTP
from core.fctp_solvers.ip_grb import get_fctp_bfs
from core.fctp_solvers.ip_grb import fctp_subset_connections
from core.fctp_solvers.ip_grb import capacitated_fctp_subset_connections
from core.fctp_solvers.ip_grb import fixed_step_fctp_subset_connections
from core.fctp_solvers.ip_grb import capacitated_fctp
from core.fctp_solvers.ip_grb import sol_vals
from core.fctp_solvers.heuristics import lcm_fctp
from core.utils.preprocessing import get_subproblems
from core.utils.postprocessing import reindex_sol


def get_max_likelihood_sol(instance, predictions):
    """Get feasibile solution that greedily maximizes the total edge likelihood.

    FCTP: Apply Least-Cost Method using negative predictions as costs.
    C-FCTP: Solve Fixed-Charge Problem using inverse predictions as fixed costs.

    Parameters
    ----------
    instance: FCTP
        FCTP instance.
    predictions: 2D np.array
        Prediction matrix.

    Returns
    -------
    sol: dict
        Greedy max-likelihood solution.

    """
    if isinstance(instance, CapacitatedFCTP):
        m, x, _ = capacitated_fctp(
            supplies=instance.supply,
            demands=instance.demand,
            var_costs=np.zeros_like(predictions, dtype=int),
            fix_costs=1 / predictions,
            edge_capacities=instance.edge_capacities,
        )
        m.setParam("OutputFlag", 0)
        m.setParam("TimeLimit", 10)
        m.setParam("Seed", 0)
        m.optimize()
        sol = sol_vals(x)
    else:
        sol = lcm_fctp(instance, costs=-predictions)
    return sol


def sol_edge_predictor_wrapper(instance, predictor_model):
    """Wrapper function for solution edge prediction.

    Parameters
    ----------
    instance: FCTP
        FCTP instance.
    predictor_model: BaseSolEdgePredictor
        Solution edge predictor.

    Returns
    -------
    predictions: 2D np.array
        Prediction matrix.

    """
    predictor_model, feature_fun = predictor_model

    inputs = tuple(feature_fun(instance))
    inputs = tuple([torch.tensor(input) for input in inputs])

    predictions = predictor_model.predict_edges(inputs, train=False)
    if isinstance(predictions, tuple):
        _, predictions = predictions

    predictions = predictions.detach().numpy()

    return predictions.reshape(instance.m, instance.n)


def get_reduced_problem(
    instance,
    predictor_model,
    threshold_type="size",
    threshold=0.5,
):
    """Wrapper function to get reduced problem.

    Step 1: Make predictions
    Step 2: Select edges (incl. feasibiliyt edges)

    Parameters
    ----------
    instance: FCTP
        FCTP instance.
    predictor_model: BaseSolEdgePredictor
        Solution edge predictor.
    threshold_type: str, optional
        Type of threshold. Can be 'size' or 'prob'. Default is 'size'.
    threshold: float, optional
        Threshold value. Default is 0.5.

    Returns
    -------
    relevant_connections: 2D np.array
        Binary edge mask indicating selected edges
    tuple of int:
        Number of predicted edges (without feasibility edges) and selected edges.

    """
    # get edge values/likelihoods
    edge_likelihood = sol_edge_predictor_wrapper(instance, predictor_model)

    # select the most likely edges
    if threshold_type == "size":
        threshold = np.quantile(edge_likelihood, 1 - threshold)
    relevant_connections = edge_likelihood >= threshold

    num_edges_pred = np.sum(relevant_connections)

    # add heuristic solution to set of edges to guarantee feasibility
    greedy_sol = get_max_likelihood_sol(instance, edge_likelihood)
    for (i, j), val in greedy_sol.items():
        if val > 0:
            relevant_connections[i, j] = True

    num_edges_enriched = np.sum(relevant_connections)

    return relevant_connections, (num_edges_pred, num_edges_enriched)


def solve_reduced_problem(
    instance,
    relevant_connections,
    decoder="exact",
    decoder_cfg=None,
    decoder_env=None,
    seed=0,
):
    """Wrapper function to solve reduced problem.

    Parameters
    ----------
    instance: FCTP
        FCTP instance.
    relevant_connections: 2D np.array
        Binary edge mask indicating selected edges
    decoder: str, optional
        Solver to use. Can be 'exact', 'lp', 'ts' or 'ea'. Default is 'exact'.
    decoder_cfg: dict, optional
        Decoder config.
    decoder_env: optional
        Decoder environment (for TS or EA).
    seed: int, optional
        Solver seed. Default is 0.

    Returns
    -------
    solution: dict
        Solution dictionary.
    runtime: int or float
        Solver runtime.
    status: int or str
        Solver status code.
    mip_gap: float
        MIP gap if applicable.

    """
    if decoder_cfg is None:
        decoder_cfg = {}

    # solve reduced FCTP
    status = None
    mip_gap = None
    if decoder in ["exact", "lp"]:
        if isinstance(instance, CapacitatedFCTP):
            model, x, _ = capacitated_fctp_subset_connections(
                instance.supply,
                instance.demand,
                instance.var_costs,
                instance.fix_costs,
                instance.edge_capacities,
                relevant_connections,
                relax=(decoder == "lp"),
            )
        elif isinstance(instance, FixedStepFCTP):
            model, x, _ = fixed_step_fctp_subset_connections(
                instance.supply,
                instance.demand,
                instance.var_costs,
                instance.fix_costs,
                instance.vehicle_capacities,
                relevant_connections,
                relax=(decoder == "lp"),
            )
        else:
            model, x, _ = fctp_subset_connections(
                instance.supply,
                instance.demand,
                instance.var_costs,
                instance.fix_costs,
                relevant_connections,
                relax=(decoder == "lp"),
            )
        model.setParam("OutputFlag", 0)
        if decoder_cfg.get("grb_timeout") is not None:
            model.setParam("TimeLimit", decoder_cfg["grb_timeout"])
        if decoder_cfg.get("grb_threads") is not None:
            model.setParam("Threads", decoder_cfg.get("grb_threads"))
        model.setParam("Seed", seed)
        model.optimize()
        solution = sol_vals(x)
        runtime = model.Runtime
        status = model.Status
        if decoder == "exact":
            mip_gap = model.MIPGap
    elif decoder == "ts":
        if isinstance(instance, CapacitatedFCTP) or isinstance(instance, FixedStepFCTP):
            raise NotImplementedError
        start = time.time()
        subproblems = get_subproblems(instance, relevant_connections)
        if len(subproblems) == 1:
            bfs = get_fctp_bfs(instance, edge_mask=relevant_connections)
            solution, _, _ = decoder_env.run(
                instance,
                bfs,
                decoder_cfg,
                relevant_connections,
            )
        else:
            sub_sols = []
            for subproblem in subproblems:
                (
                    s_nodes,
                    d_nodes,
                    sub_instance,
                    sub_conns,
                ) = subproblem
                if len(s_nodes) == 1:
                    sub_sol = {(0, j): d for j, d in enumerate(sub_instance.demand)}
                elif len(d_nodes) == 1:
                    sub_sol = {(i, 0): s for i, s in enumerate(sub_instance.supply)}
                else:
                    bfs = get_fctp_bfs(
                        sub_instance,
                        edge_mask=sub_conns,
                    )
                    sub_sol, _, _ = decoder_env.run(
                        sub_instance,
                        bfs,
                        decoder_cfg,
                        sub_conns,
                    )
                sub_sols.append(reindex_sol(sub_sol, s_nodes, d_nodes))
            solution = {k: v for sub_sol in sub_sols for (k, v) in sub_sol.items()}
        runtime = time.time() - start
    elif decoder == "ea":
        if isinstance(instance, CapacitatedFCTP) or isinstance(instance, FixedStepFCTP):
            raise NotImplementedError
        start = time.time()
        subproblems = get_subproblems(instance, relevant_connections)
        if len(subproblems) == 1:
            solution, _, _ = decoder_env.run(
                instance,
                decoder_cfg,
                relevant_connections,
            )
        else:
            sub_sols = []
            for subproblem in subproblems:
                (
                    s_nodes,
                    d_nodes,
                    sub_instance,
                    sub_conns,
                ) = subproblem
                if len(s_nodes) == 1:
                    sub_sol = {(0, j): d for j, d in enumerate(sub_instance.demand)}
                elif len(d_nodes) == 1:
                    sub_sol = {(i, 0): s for i, s in enumerate(sub_instance.supply)}
                else:
                    sub_sol, _, _ = decoder_env.run(
                        sub_instance,
                        decoder_cfg,
                        sub_conns,
                    )
                sub_sols.append(reindex_sol(sub_sol, s_nodes, d_nodes))
            solution = {k: v for sub_sol in sub_sols for (k, v) in sub_sol.items()}
        runtime = time.time() - start
    else:
        raise ValueError
    return solution, runtime, status, mip_gap


def ml_based_fctp_reduction(
    instance,
    predictor_model,
    threshold_type="size",
    threshold=0.5,
    decoder="exact",
    decoder_cfg=None,
    decoder_env=None,
    seed=0,
):
    """Wrapper function for ML-based reduce-then-optimize.

    Step 1: Get reduced instance
    Step 2: Solve reduced instance

    Parameters
    ----------
    instance: FCTP
        FCTP instance.
    predictor_model: BaseSolEdgePredictor
        Solution edge predictor.
    threshold_type: str, optional
        Type of threshold. Can be 'size' or 'prob'. Default is 'size'.
    threshold: float, optional
        Threshold value. Default is 0.5.
    decoder: str, optional
        Solver to use. Can be 'exact', 'lp', 'ts' or 'ea'. Default is 'exact'.
    decoder_cfg: dict, optional
        Decoder config.
    decoder_env: optional
        Decoder environment (for TS or EA).
    seed: int, optional
        Solver seed. Default is 0.

    Returns
    -------
    solution: dict
        Solution dictionary.
    num_edges_pred:
        Number of predicted edges (without feasibility edges)
    num_edges_enriched:
        Number of selected edges.
    runtime: int or float
        Solver runtime.
    status: int or str
        Solver status code.
    mip_gap: float
        MIP gap if applicable.

    """

    relevant_connections, (num_edges_pred, num_edges_enriched) = get_reduced_problem(
        instance,
        predictor_model,
        threshold_type,
        threshold,
    )

    solution, runtime, status, mip_gap = solve_reduced_problem(
        instance,
        relevant_connections,
        decoder,
        decoder_cfg,
        decoder_env,
        seed,
    )

    return solution, num_edges_pred, num_edges_enriched, runtime, status, mip_gap
