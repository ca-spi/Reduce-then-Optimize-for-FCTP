""" Plot optimality gap and MIP gap across edge set sizes for BASE_120x120 (GNN+GRB) (Figure 6). """

import os

from core.evaluation.benchmarking_utils import get_optgap_and_mipgap_plot
from core.utils.visualization import warmstart_px_save


if __name__ == "__main__":

    out_dir = "results/benchmarking_performance"
    os.makedirs(out_dir, exist_ok=True)

    benchmarking_dir = (
        "benchmarking_paper/fctp_agarwal-aneja_120_120_B20_Theta0.2_BF1.0"
    )
    experiment_dir = os.path.join(
        benchmarking_dir,
        "ml-reduction",
        "size",
        "exact-grb_timeout_7200",
        "model_gcnn_features_bipartite_raw_prediction_task_binary_classification_normalization_standard_hidden_layer_dim_20_num_conv_layers_10_num_dense_layers_2",
    )
    baseline = "exact/grb_timeout_43200"

    fig = get_optgap_and_mipgap_plot(
        benchmarking_dir=benchmarking_dir,
        experiment_dir=experiment_dir,
        baseline=baseline,
        xaxis_title="Size threshold",
        yaxis_title=("Performance gap [%]", "MIP gap [s]"),
        xaxis_reversed=False,
    )
    filename = "opt_gap_vs_mip_gap_size_threshold_120x120.pdf"
    img_path = os.path.join(out_dir, filename)
    warmstart_px_save(img_path)
    fig.write_image(img_path)
