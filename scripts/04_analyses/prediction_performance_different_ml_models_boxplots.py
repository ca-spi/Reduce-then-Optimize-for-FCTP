""" Compare different edge prediction models in terms of ML KPIs (Figure 4). """

import os

from core.evaluation.ml_model_evaluation import (
    prediction_performance_comparison_boxplot,
)
from core.utils.visualization import warmstart_px_save


if __name__ == "__main__":

    model_dir = "trained_models/sol_edge_predictor"
    out_dir = "results/prediction_performance"
    os.makedirs(out_dir, exist_ok=True)

    checkpoints = {
        "LR - basic": "linear_logreg/model_linear_logreg_features_combined_raw_edge_features_prediction_task_binary_classification_normalization_standard",
        "LR - adv.": "linear_logreg/model_linear_logreg_features_advanced_edge_features_prediction_task_binary_classification_normalization_standard",
        "LR - adv. + stat.": "linear_logreg/model_linear_logreg_features_advanced_edge_features_plus_stat_features_prediction_task_binary_classification_normalization_standard",
        "MLP - basic": "edge_mlp/model_edge_mlp_features_combined_raw_edge_features_prediction_task_binary_classification_normalization_standard_hidden_layer_dim_20_num_dense_layers_5",
        "MLP - adv.": "edge_mlp/model_edge_mlp_features_advanced_edge_features_prediction_task_binary_classification_normalization_standard_hidden_layer_dim_20_num_dense_layers_5",
        "MLP - adv. + stat.": "edge_mlp/model_edge_mlp_features_advanced_edge_features_plus_stat_features_prediction_task_binary_classification_normalization_standard_hidden_layer_dim_20_num_dense_layers_5",
        "GNN": "gcnn/model_gcnn_features_bipartite_raw_prediction_task_binary_classification_normalization_standard_hidden_layer_dim_20_num_conv_layers_10_num_dense_layers_2",
        "GNN - adv.": "gcnn/model_gcnn_features_bipartite_advanced_edge_features_prediction_task_binary_classification_normalization_standard_hidden_layer_dim_20_num_conv_layers_10_num_dense_layers_2",
        "GNN - adv. + stat.": "gcnn/model_gcnn_features_bipartite_advanced_edge_features_plus_stat_features_prediction_task_binary_classification_normalization_standard_hidden_layer_dim_20_num_conv_layers_10_num_dense_layers_2",
    }

    checkpoints = {
        k: os.path.join(
            model_dir, "fctp_agarwal-aneja_15_15_B20_Theta0.2_BF1.0", v, "cross_val"
        )
        for k, v in checkpoints.items()
    }

    metrics = ["fscore"]  # ["accuracy", "recall", "precision", "fscore", "loss"]
    yaxis_titles = {c: c.capitalize() for c in metrics}
    yaxis_titles["fscore"] = "F\u2082"

    for metric in metrics:
        fig = prediction_performance_comparison_boxplot(
            checkpoints=checkpoints,
            metric=metric,
            yaxis_title=yaxis_titles[metric],
        )
        filename = f"prediction_performance_comparison_{metric}.pdf"
        img_path = os.path.join(out_dir, filename)
        warmstart_px_save(img_path)
        fig.write_image(img_path)
