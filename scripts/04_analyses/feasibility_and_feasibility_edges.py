""" Analyze feasibility of predictions and usage of feasibiliyt edges (Table 12, Figure 15). """

import os
import gzip
import os
import pickle as pkl

import numpy as np
import pandas as pd

from core.evaluation.benchmarking_utils import (
    get_num_edges_pred_vs_used_barplot,
)
from core.data_processing.data_utils import load_instance
from core.utils.ml_utils import load_edge_predictor_model
from core.ml_models.wrapper import (
    sol_edge_predictor_wrapper,
)
from core.fctp_solvers.ip_grb import fctp_subset_connections
from core.utils.visualization import warmstart_px_save
from core.utils.postprocessing import sol_to_matrix


if __name__ == "__main__":
    instance_root_dir = "data_paper/instances/benchmarking"
    benchmarking_dir = "benchmarking_paper"
    result_dir = "results/prediction_performance"
    model_dir = "trained_models/sol_edge_predictor"
    model_spec = "gcnn/model_gcnn_features_bipartite_raw_prediction_task_binary_classification_normalization_standard_hidden_layer_dim_20_num_conv_layers_10_num_dense_layers_2"

    os.makedirs(result_dir, exist_ok=True)

    dataset = "fctp_agarwal-aneja_15_15_B20_Theta0.2_BF1.0"
    instance_dir = os.path.join(instance_root_dir, dataset)
    model_path = os.path.join(
        model_dir, dataset, model_spec, "cross_val", "best_checkpoint.pth.tar"
    )
    experiment_dir = os.path.join(
        benchmarking_dir,
        dataset,
        "ml-reduction",
        "size",
        "exact-grb_timeout_43200",
        model_spec.split("/")[-1],
    )

    ##############################################
    # Barplot #edges predicted vs. #edges incl. feasibility edges
    ##############################################
    fig = get_num_edges_pred_vs_used_barplot(
        benchmarking_dir=os.path.join(benchmarking_dir, dataset),
        experiment_dir=experiment_dir,
        baseline="exact/grb_timeout_43200",
    )
    filename = "nedges_pred_vs_used.pdf"
    img_path = os.path.join(result_dir, filename)
    warmstart_px_save(img_path)
    fig.write_image(img_path)

    ##############################################
    # Feasibility and usage feasibility edges in final solutions
    ##############################################
    relevant_edgesetsizes = [float(i) for i in os.listdir(experiment_dir)]
    instances = os.listdir(instance_dir)
    predictor_model = load_edge_predictor_model(model_path)

    usage_feasibility_edges = {s: [] for s in relevant_edgesetsizes}
    edge_set_feasibility = {s: [] for s in relevant_edgesetsizes}

    for instance_name in instances:
        instance_id = instance_name.split(".")[0].split("_")[-1]
        print(f"Evaluating instance {instance_id}...")
        solution_filename = f"sol_instance_{instance_id}.pkl.gz"

        # Load instance
        instance_path = os.path.join(instance_dir, instance_name)
        instance = load_instance(instance_path)

        # Get edge prediction and selection
        edge_likelihood = sol_edge_predictor_wrapper(instance, predictor_model)

        for edgesetsize in relevant_edgesetsizes:
            print(f"...with edge set size {edgesetsize}")
            # Edge selection
            threshold = np.quantile(edge_likelihood, 1 - edgesetsize)
            edges_selected = np.where(edge_likelihood >= threshold, 1, 0)

            # Load solution
            solution_path = os.path.join(
                experiment_dir, str(edgesetsize), solution_filename
            )
            with gzip.open(solution_path, "rb") as f:
                result_dict = pkl.load(f)
            solution = result_dict["solution"]
            sol_adj_matrix = np.where(
                sol_to_matrix(solution, edges_selected.shape) > 0, 1, 0
            )

            # Compare solution to predicted edges
            usage_feasibility_edges[edgesetsize].append(
                (sol_adj_matrix > edges_selected).astype(int).sum()
            )

            # Check feasibility of predicted/selected edge set
            m, _, _ = fctp_subset_connections(
                instance.supply,
                instance.demand,
                instance.var_costs,
                instance.fix_costs,
                edges_selected,
                relax=True,
            )
            m.optimize()
            if m.status == 2:
                edge_set_feasibility[edgesetsize].append(True)
            else:
                edge_set_feasibility[edgesetsize].append(False)
                print(m.status)

    avg_num_feasibility_edges = {
        k: np.round(np.mean(v), decimals=2) for k, v in usage_feasibility_edges.items()
    }
    num_instances_using_feasibility_edges = {
        k: np.sum(np.where(np.array(v) > 0, 1, 0))
        for k, v in usage_feasibility_edges.items()
    }

    usage_feasibility_edges_df = (
        pd.DataFrame(
            [num_instances_using_feasibility_edges, avg_num_feasibility_edges],
            index=["#instances", "#edges"],
        )
        .transpose()
        .sort_index()
    )
    usage_feasibility_edges_df["#instances"] = usage_feasibility_edges_df[
        "#instances"
    ].astype(int)
    print(usage_feasibility_edges_df.transpose().to_latex())
