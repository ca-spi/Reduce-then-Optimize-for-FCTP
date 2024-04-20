""" Compare different reduction methods (Figure 11). """

import os

from core.evaluation.benchmarking_utils import (
    get_summary_performance_plot_across_edgesetsizes,
)
from core.utils.visualization import warmstart_px_save


if __name__ == "__main__":

    benchmarking_dir = "benchmarking_paper"
    out_dir = "results/benchmarking_performance"
    os.makedirs(out_dir, exist_ok=True)

    grb_rt = {15: 43200, 30: 7200}
    basline_rt = {15: 43200, 30: 43200}

    relevant_edgesetsizes = {
        15: [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
        30: [0.1, 0.15, 0.2, 0.25, 0.3],
    }

    for ps in grb_rt.keys():
        print(f"Evaluating {ps}x{ps} instances")
        benchmarking_dir = os.path.join(
            benchmarking_dir, f"fctp_agarwal-aneja_{ps}_{ps}_B20_Theta0.2_BF1.0"
        )

        experiment_dirs = {
            "ML - GNN": f"ml-reduction/size/exact-grb_timeout_{grb_rt[ps]}/model_gcnn_features_bipartite_raw_prediction_task_binary_classification_normalization_standard_hidden_layer_dim_20_num_conv_layers_10_num_dense_layers_2",
            # "ML - MLP": f"ml-reduction/size/exact-grb_timeout_{grb_rt[ps]}/model_edge_mlp_features_advanced_edge_features_prediction_task_binary_classification_normalization_standard_hidden_layer_dim_20_num_dense_layers_5",
            "ML - MLP": f"ml-reduction/size/exact-grb_timeout_{grb_rt[ps]}/model_edge_mlp_features_advanced_edge_features_plus_stat_features_prediction_task_binary_classification_normalization_standard_hidden_layer_dim_20_num_dense_layers_5",
            # "ML - LR": f"ml-reduction/size/exact-grb_timeout_{grb_rt[ps]}/model_linear_logreg_features_advanced_edge_features_prediction_task_binary_classification_normalization_standard",
            "ML - LR": f"ml-reduction/size/exact-grb_timeout_{grb_rt[ps]}/model_linear_logreg_features_advanced_edge_features_plus_stat_features_prediction_task_binary_classification_normalization_standard",
            "\u03BA-random-edges": f"k-random-edges/exact-grb_timeout_{grb_rt[ps]}",
            "\u03BA-shortest-edges": f"k-shortest-edges/exact-grb_timeout_{grb_rt[ps]}",
        }
        experiment_dirs = {
            k: os.path.join(benchmarking_dir, v) for k, v in experiment_dirs.items()
        }

        baseline = f"exact/grb_timeout_{basline_rt[ps]}"

        # Mean optimality gap
        fig = get_summary_performance_plot_across_edgesetsizes(
            benchmarking_dir=benchmarking_dir,
            experiment_dirs=experiment_dirs,
            baseline=baseline,
            metric="mean",
            relevant_edgesetsizes=relevant_edgesetsizes[ps],
            legend_title=None,
            legend_right=True,
            xaxis_title="Size threshold",
            yaxis_title="Optimality gap [%]",
        )
        filename = f"opt_gap_across_edgesetsizes_edge_selectors_mean_{ps}x{ps}.pdf"
        img_path = os.path.join(out_dir, filename)
        warmstart_px_save(img_path)
        fig.write_image(img_path)

        # Number of instances solved to optimality
        fig = get_summary_performance_plot_across_edgesetsizes(
            benchmarking_dir=benchmarking_dir,
            experiment_dirs=experiment_dirs,
            baseline=baseline,
            metric="percentage optimally solved",
            relevant_edgesetsizes=relevant_edgesetsizes[ps],
            legend_title=None,
            xaxis_title="Size threshold",
            yaxis_title="Instances solved to optimality [%]",
            legend_right=False,
        )
        filename = f"opt_solved_across_edgesetsizes_edge_selectors_{ps}x{ps}.pdf"
        img_path = os.path.join(out_dir, filename)
        warmstart_px_save(img_path)
        fig.write_image(img_path)
