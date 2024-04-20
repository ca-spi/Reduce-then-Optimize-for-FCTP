""" Visualize reduce-then-optimize pipeline (Figure 1). """

import os

import numpy as np

from core.data_processing.data_utils import load_instance
from core.fctp_solvers.ip_grb import fctp
from core.fctp_solvers.ip_grb import fctp_subset_connections
from core.fctp_solvers.ip_grb import sol_vals
from core.ml_models.wrapper import get_max_likelihood_sol
from core.ml_models.wrapper import sol_edge_predictor_wrapper
from core.utils.ml_utils import load_edge_predictor_model
from core.utils.postprocessing import sol_to_matrix
from core.utils.visualization import (
    visual_evaluation_euclidian_sol_edge_prediction,
)


def get_instance_matrices(instance, prediction_model, threshold_quantile=0.3):
    """Get edge matrices representing instances and solutions.

    Five matrices:
    1. Edge predictions
    2. Predicted edge set
    3. Predicted edge set + feasibility edges
    4. Solution
    5. Optimal solution for comparison

    Parameters
    ----------
    instance: FCTP
        FCTP instance of interest.
    prediction_model: BaseSolEdgePredictor
        Solution edge predictor.
    threshold_quantile: float, optional
        Threshold value. Default is 0.3.

    Returns
    -------
    tuple:
        5-element tuple containing teh five edge matrices.
    float:
        Optimality gap.

    """
    supply, demand, var_costs, fix_costs = (
        instance.supply,
        instance.demand,
        instance.var_costs,
        instance.fix_costs,
    )
    # get optimal solution
    model, x, _ = fctp(supply, demand, var_costs, fix_costs)
    model.optimize()
    opt_sol = sol_to_matrix(sol_vals(x))
    opt_sol_val = model.ObjVal

    # get predictions
    predictions = sol_edge_predictor_wrapper(instance, prediction_model)

    # select the most likely edges
    threshold = np.quantile(predictions, 1 - threshold_quantile)
    relevant_connections = np.where(predictions > threshold, 1, 0)
    print(np.sum(relevant_connections))

    # add heuristic solution to set of edges to guarantee feasibility
    greedy_sol = get_max_likelihood_sol(instance, predictions)
    relevant_connections_plus = relevant_connections.copy()
    for (i, j), val in greedy_sol.items():
        if val > 0:
            relevant_connections_plus[i, j] = True

    # get solution
    model, x, _ = fctp_subset_connections(
        supply, demand, var_costs, fix_costs, relevant_connections_plus
    )
    model.optimize()
    reduced_sol = sol_to_matrix(sol_vals(x))
    opt_gap = np.round(100 * (model.ObjVal / opt_sol_val - 1), 2)

    return [
        predictions,
        relevant_connections,
        relevant_connections_plus,
        reduced_sol,
        opt_sol,
    ], opt_gap


if __name__ == "__main__":

    # set up paths
    result_dir = "results/visualization"
    instance_root_dir = "data_paper/instances/benchmarking"
    model_dir = "trained_models/sol_edge_predictor"
    model_spec = "gcnn/model_gcnn_features_bipartite_raw_prediction_task_binary_classification_normalization_standard_hidden_layer_dim_20_num_conv_layers_10_num_dense_layers_2"

    os.makedirs(result_dir, exist_ok=True)

    dataset = "fctp_euclidian_15_15_B20_Theta0.2_BF1.0"
    instance_dir = os.path.join(instance_root_dir, dataset)
    model_path = os.path.join(
        model_dir, dataset, model_spec, "cross_val", "best_checkpoint.pth.tar"
    )

    # select instances
    num_instances = 10
    instance_paths = [
        os.path.join(instance_dir, p) for p in os.listdir(instance_dir)[:num_instances]
    ]

    # load model
    predictor_model = load_edge_predictor_model(model_path)
    threshold_quantile = 0.3

    # visualize exemplary instances
    for i, instance_path in enumerate(instance_paths):
        instance = load_instance(instance_path)
        matrices, opt_gap = get_instance_matrices(
            instance, predictor_model, threshold_quantile=threshold_quantile
        )
        matrices = [matrices[i] for i in [0, 2, 3, 4]]
        # visualize performance
        subtitles = [
            "Edge Predictions",
            "Reduced Edge Set",
            "Solution",
            "Optimal Solution",
        ]
        fig, _ = visual_evaluation_euclidian_sol_edge_prediction(
            instance.supplier_locations,
            instance.customer_locations,
            matrices,
            title=None,
            text=f"Optimality gap: {opt_gap}%",
            subtitles=subtitles,
        )
        fig.savefig(os.path.join(result_dir, f"reduce_then_optimize_pipeline_{i}.pdf"))
