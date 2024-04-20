""" Plot optimality gaps for different reduction levels and time limits (GNN+GRB) (Figure 8). """

import os

from core.evaluation.benchmarking_utils import \
    get_summary_performance_plot_across_edgesetsizes
from core.utils.visualization import warmstart_px_save

if __name__ == "__main__":

    root_benchmarking_dir = "benchmarking_paper"
    out_dir = "results/benchmarking_performance"
    os.makedirs(out_dir, exist_ok=True)

    timelimits = {30: [5, 10, 30, 60], 120: [10, 30, 60, 300]}
    relevant_edgeset_sizes = {
        30: [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
        120: [0.025, 0.05, 0.1, 0.15, 0.2, 0.25],
    }

    for ps, timelimit_ps in timelimits.items():
        print(f"Evaluating {ps}x{ps} instances")
        dataset = f"fctp_agarwal-aneja_{ps}_{ps}_B20_Theta0.2_BF1.0"
        experiment_dirs = {
            tl: f"ml-reduction/size/exact-grb_timeout_{tl}/model_gcnn_features_bipartite_raw_prediction_task_binary_classification_normalization_standard_hidden_layer_dim_20_num_conv_layers_10_num_dense_layers_2"
            for tl in timelimit_ps
        }
        experiment_dirs = {
            k: os.path.join(root_benchmarking_dir, dataset, v)
            for k, v in experiment_dirs.items()
        }
        baselines = {tl: f"exact/grb_timeout_{tl}" for tl in timelimit_ps}

        for metric in ["mean"]:
            fig = get_summary_performance_plot_across_edgesetsizes(
                benchmarking_dir=os.path.join(root_benchmarking_dir, dataset),
                experiment_dirs=experiment_dirs,
                baseline=baselines,
                relevant_edgesetsizes=relevant_edgeset_sizes[ps],
                metric=metric,
                xaxis_title="Size threshold",
                yaxis_title="Performance gap [%]",
                legend_title="Runtime limit [s]",
                hline=0,
            )
            filename = f"opt_gap_across_edgesetsizes_for_diff_timelimits_{ps}x{ps}_{metric}.pdf"
            img_path = os.path.join(out_dir, filename)
            warmstart_px_save(img_path)
            fig.write_image(img_path)
