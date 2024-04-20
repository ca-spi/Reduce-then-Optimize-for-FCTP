""" Analyze characteristics of challenging instances (Table 10, Figure 12). """

import os

import gurobipy as gp
import numpy as np
import pandas as pd
import plotly.express as px
from scipy import stats

from core.data_processing.data_utils import load_instance
from core.utils.ml_utils import load_edge_predictor_model
from core.ml_models.wrapper import (
    sol_edge_predictor_wrapper,
)
from core.ml_models.wrapper import (
    ml_based_fctp_reduction,
)
from core.fctp_solvers.ip_grb import fctp
from core.utils.kpi import get_accuracy
from core.utils.kpi import get_recall
from core.utils.visualization import warmstart_px_save

if __name__ == "__main__":
    result_dir = "results/benchmarking_performance"
    instance_root_dir = "data_paper/instances/benchmarking"
    model_dir = "trained_models/sol_edge_predictor"
    model_spec = "gcnn/model_gcnn_features_bipartite_raw_prediction_task_binary_classification_normalization_standard_hidden_layer_dim_20_num_conv_layers_10_num_dense_layers_2"

    os.makedirs(result_dir, exist_ok=True)

    dataset = "fctp_agarwal-aneja_15_15_B20_Theta0.2_BF1.0"
    instance_dir = os.path.join(instance_root_dir, dataset)
    model_path = os.path.join(
        model_dir, dataset, model_spec, "cross_val", "best_checkpoint.pth.tar"
    )

    predictor_model = load_edge_predictor_model(model_path)

    instances = os.listdir(instance_dir)
    std_probability = []
    avg_probability = []
    accuracy = []
    recall = []
    avg_sol_edge_probability = []
    avg_rank = []
    num_edges_needed_to_cover_sol = []
    opt_gaps = {0.4: [], 0.3: [], 0.2: []}
    gap_x_best = {2: [], 3: [], 4: [], 5: []}
    runtime = []
    ts_opt_gap = []
    ea_opt_gap = []
    opt_vals = []

    for idx, instance_name in enumerate(instances):
        instance_id = instance_name.split(".")[0].split("_")[-1]
        print(f"Evaluating instance {instance_id}...")

        # Load instance
        instance_path = os.path.join(instance_dir, instance_name)
        instance = load_instance(instance_path)

        # Solve
        model, x, y = fctp(
            instance.supply, instance.demand, instance.var_costs, instance.fix_costs
        )
        model.setParam("OutputFlag", 0)
        model.setParam("Seed", 0)
        model.setParam("TimeLimit", 600)
        model.optimize()
        print(f"Model status: {model.status}")
        opt_val = model.ObjVal
        opt_vals.append(opt_val)
        print(f"Opt val: {opt_val}")
        runtime.append(model.Runtime)

        # Get edge predictions
        edge_likelihood = sol_edge_predictor_wrapper(instance, predictor_model)
        avg_probability.append(np.mean(edge_likelihood))
        std_probability.append(np.std(edge_likelihood))

        num_edges = int(len(model.getVars()) / 2)
        edge_index = np.unravel_index(np.arange(0, num_edges), instance.var_costs.shape)
        edge_index = [(edge_index[0][i], edge_index[1][i]) for i in range(num_edges)]

        # Retrieve solution
        sol = np.round(model.X).astype(int)[num_edges:]  # only y
        edges_sol = [edge_index[i] for i, v in enumerate(sol) if v > 0]

        # Evaluate accuracy
        target = np.zeros_like(edge_likelihood)
        for e in edges_sol:
            target[e] = 1
        accuracy.append(get_accuracy(np.round(edge_likelihood), target))
        recall.append(get_recall(np.round(edge_likelihood), target))

        # Evaluate support
        support_for_x = [edge_likelihood[e] for e in edges_sol]
        print(
            f"Average probability of solution edges: {np.mean(support_for_x)} (overall mean: {np.mean(edge_likelihood)})"
        )
        avg_sol_edge_probability.append(np.mean(support_for_x))

        # Get edge ranking
        ranking = np.unravel_index(
            np.flip(np.argsort(edge_likelihood.flatten())), edge_likelihood.shape
        )
        edge_ranks = np.zeros(edge_likelihood.shape, dtype=int)
        for i in range(len(ranking[0])):
            edge_ranks[ranking[0][i], ranking[1][i]] = i + 1

        ranks_x = [edge_ranks[e] for e in edges_sol]
        print(f"Average rank of solution edges: {np.mean(ranks_x)}")
        print(
            f"Number of edges needed to cover solution (= max rank): {np.max(ranks_x)}"
        )
        avg_rank.append(np.mean(ranks_x))
        num_edges_needed_to_cover_sol.append(np.max(ranks_x))

        for thrsh in opt_gaps.keys():
            # Evaluate reduce-then-optimize
            (
                sol,
                _,
                _,
                _,
                _,
                _,
            ) = ml_based_fctp_reduction(
                instance,
                predictor_model=predictor_model,
                threshold_type="size",
                threshold=thrsh,
                decoder="exact",
            )
            sol_val = instance.eval_sol_dict(sol)
            opt_gaps[thrsh].append(sol_val / opt_val - 1)

        # Gap to x-best
        for i in [2, 3, 4, 5]:
            model.addConstr(
                gp.quicksum(y[e] for e in edges_sol) <= len(edges_sol) - 1,
                name=f"sol_cut_{i}",
            )
            model.optimize()
            print(f"Model status: {model.status}")
            second_best_val = model.ObjVal
            gap = (second_best_val / opt_val) - 1
            print(f"{i}-best val: {second_best_val} (Gap: {np.round(gap*100,3)}%)")
            gap_x_best[i].append(gap)
            sol = np.round(model.X).astype(int)[num_edges:]  # only y
            edges_sol = [edge_index[i] for i, v in enumerate(sol) if v > 0]

    # ############################################
    # # Evaluation
    # ############################################
    challenging_idx = np.flip(np.argsort(opt_gaps[0.3])[-5:])

    challenging_instances = [instances[i] for i in challenging_idx]

    performance = {
        instances[idx]: {
            "opt gap 40%": np.round(opt_gaps[0.4][idx] * 100, 2),
            "opt gap 30%": np.round(opt_gaps[0.3][idx] * 100, 2),
            "opt gap 20%": np.round(opt_gaps[0.2][idx] * 100, 2),
            "cover": num_edges_needed_to_cover_sol[idx],
            "support opt sol": np.round(avg_sol_edge_probability[idx] * 100, 2),
            "accuracy": np.round(accuracy[idx] * 100, 2),
            "recall": np.round(recall[idx] * 100, 2),
            "gap 2nd best": np.round(gap_x_best[2][idx] * 100, 3),
            "gap 3rd best": np.round(gap_x_best[3][idx] * 100, 3),
            "gap 5th best": np.round(gap_x_best[5][idx] * 100, 3),
            "runtime": np.round(runtime[idx], 3),
        }
        for idx in challenging_idx
    }
    performance["mean"] = {
        "opt gap 40%": np.round(np.mean(opt_gaps[0.4]) * 100, 2),
        "opt gap 30%": np.round(np.mean(opt_gaps[0.3]) * 100, 2),
        "opt gap 20%": np.round(np.mean(opt_gaps[0.2]) * 100, 2),
        "cover": np.mean(num_edges_needed_to_cover_sol),
        "support opt sol": np.round(np.mean(avg_sol_edge_probability) * 100, 2),
        "accuracy": np.round(np.mean(accuracy) * 100, 2),
        "recall": np.round(np.mean(recall) * 100, 2),
        "gap 2nd best": np.round(np.mean(gap_x_best[2]) * 100, 3),
        "gap 3rd best": np.round(np.mean(gap_x_best[3]) * 100, 3),
        "gap 5th best": np.round(np.mean(gap_x_best[5]) * 100, 3),
        "runtime": np.round(np.mean(runtime), 3),
    }

    df = pd.DataFrame.from_dict(performance, orient="index")

    with pd.option_context("max_colwidth", 1000):
        print(df.to_latex())

    # Analyze correlations
    x = [
        accuracy,
        recall,
        avg_sol_edge_probability,
        num_edges_needed_to_cover_sol,
        gap_x_best[2],
        gap_x_best[3],
        gap_x_best[5],
    ]
    y = [opt_gaps[0.4], opt_gaps[0.3], opt_gaps[0.2]]
    corr_r = np.array([[stats.pearsonr(xi, yi).correlation for xi in x] for yi in y])
    corr_p = np.array([[stats.pearsonr(xi, yi).pvalue for xi in x] for yi in y])
    corr_r[corr_p >= 0.05] = np.nan
    fig = px.imshow(
        corr_r,
        text_auto=True,
        aspect="auto",
        labels=dict(
            x="Feature", y="Optimality gap at x% edge set size", color="Correlation"
        ),
        x=[
            "accuracy",
            "recall",
            "support",
            "cover",
            "gap 2nd-best",
            "gap 3rd-best",
            "gap 5th-best",
        ],
        y=["40%", "30%", "20%"],
    )
    fig.update_layout(
        template="simple_white",
        width=850,
        height=350,
        margin=dict(l=20, r=20, t=30, b=20),
    )
    filename = f"drivers_opt_gap_{dataset}.pdf"
    img_path = os.path.join(result_dir, filename)
    warmstart_px_save(img_path)
    fig.write_image(img_path)
