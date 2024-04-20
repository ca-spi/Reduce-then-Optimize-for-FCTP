""" Plot runtime against optimality gap for BASE_15x15 and BASE_30x30 (GNN+GRB) (Figure 5)"""

import os

from core.evaluation.benchmarking_utils import get_optgap_vs_runtime_plot
from core.evaluation.benchmarking_utils import (
    get_optgap_vs_runtime_plot_with_edgeset_barplot,
)
from core.utils.visualization import warmstart_px_save


if __name__ == "__main__":

    root_benchmarking_dir = "benchmarking_paper"
    out_dir = "results/benchmarking_performance"
    os.makedirs(out_dir, exist_ok=True)

    # Problem sizes and runtimes (to derive directory paths)
    grb_rt = {15: 43200, 30: 43200}
    baseline_rt = {15: 43200, 30: 43200}

    # Sames ranges for plotting to facilitate comparison
    opt_gap_ranges = {"size": [0, 5], "prob": [0, 20]}
    runtime_ranges = {15: [0, 8], 30: [0, 900]}
    offset = 0.05
    opt_gap_ranges = {
        k: [l - (u - l) * offset, u + (u - l) * offset]
        for k, (l, u) in opt_gap_ranges.items()
    }
    runtime_ranges = {
        k: [l - (u - l) * offset, u + (u - l) * offset]
        for k, (l, u) in runtime_ranges.items()
    }

    # Evaluate
    for ps in grb_rt.keys():
        print(f"Evaluating {ps}x{ps} instances")
        benchmarking_dir = os.path.join(
            root_benchmarking_dir, f"fctp_agarwal-aneja_{ps}_{ps}_B20_Theta0.2_BF1.0"
        )
        for thrsh in ["size", "prob"]:
            # Runtime vs. Opt Gap
            if thrsh == "size":
                experiment_dir = os.path.join(
                    benchmarking_dir,
                    "ml-reduction",
                    thrsh,
                    f"exact-grb_timeout_{grb_rt[ps]}",
                    "model_gcnn_features_bipartite_raw_prediction_task_binary_classification_normalization_standard_hidden_layer_dim_20_num_conv_layers_10_num_dense_layers_2",
                )
                fig = get_optgap_vs_runtime_plot(
                    benchmarking_dir=benchmarking_dir,
                    experiment_dir=experiment_dir,
                    baseline=f"exact/grb_timeout_{baseline_rt[ps]}",
                    xaxis_title="Size threshold",
                    yaxis_title=("Optimality gap [%]", "Runtime [s]"),
                    xaxis_reversed=False,
                    optgap_range=opt_gap_ranges[thrsh],
                    runtime_range=runtime_ranges[ps],
                )
                filename = f"opt_gap_vs_runtime_{thrsh}_threshold_{ps}x{ps}.pdf"
                img_path = os.path.join(out_dir, filename)
                warmstart_px_save(img_path)
                fig.write_image(img_path)
            # Runtime vs. Opt Gap + Problem Size
            else:
                experiment_dir = os.path.join(
                    benchmarking_dir,
                    "ml-reduction",
                    thrsh,
                    f"exact-grb_timeout_{grb_rt[ps]}",
                    "model_gcnn_features_bipartite_raw_prediction_task_binary_classification_normalization_standard_hidden_layer_dim_20_num_conv_layers_10_num_dense_layers_2",
                )
                fig = get_optgap_vs_runtime_plot_with_edgeset_barplot(
                    benchmarking_dir=benchmarking_dir,
                    experiment_dir=experiment_dir,
                    relevant_edgesetsizes=[
                        0.2,
                        0.3,
                        0.4,
                        0.5,
                        0.6,
                        0.7,
                        0.8,
                        0.9,
                    ],
                    baseline=f"exact/grb_timeout_{baseline_rt[ps]}",
                    xaxis_title="Probability threshold",
                    yaxis_title=("Optimality gap [%]", "Runtime [s]"),
                    xaxis_reversed=True,
                    optgap_range=opt_gap_ranges[thrsh],
                    runtime_range=runtime_ranges[ps],
                )
                filename = f"opt_gap_vs_runtime_{thrsh}_threshold_{ps}x{ps}.pdf"
                img_path = os.path.join(out_dir, filename)
                warmstart_px_save(img_path)
                fig.write_image(img_path)
