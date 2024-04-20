""" Report validation accuracy and recall for different datasets (Table 4). """

import os

import pandas as pd

from core.evaluation.ml_model_evaluation import (
    prediction_performance_comparison_table,
)


if __name__ == "__main__":

    model_dir = "trained_models/sol_edge_predictor"
    model_spec = "gcnn/model_gcnn_features_bipartite_raw_prediction_task_binary_classification_normalization_standard_hidden_layer_dim_20_num_conv_layers_10_num_dense_layers_2"

    checkpoints = {
        "BASE": "fctp_agarwal-aneja_15_15_B20_Theta0.2_BF1.0",
        "B50": "fctp_agarwal-aneja_15_15_B50_Theta0.2_BF1.0",
        "UB10%": "fctp_agarwal-aneja_15_15_B20_Theta0.2_BF1.1",
        "UB30%": "fctp_agarwal-aneja_15_15_B20_Theta0.2_BF1.3",
        "RBM15": "fctp_roberti_15_15_B20_Theta0.2_BF1.0",
        "EUCL": "fctp_euclidian_15_15_B20_Theta0.2_BF1.0",
        "VFCR0.5": "fctp_agarwal-aneja_15_15_B20_Theta0.5_BF1.0",
        "VFCR0.0": "fctp_agarwal-aneja_15_15_B20_Theta0.0_BF1.0",
        "C-FCTP": "c-fctp_agarwal-aneja_15_15_B20_Theta0.2_BF1.0",
        "FS-FCTP": "fs-fctp_agarwal-aneja_15_15_B20_Theta0.2_BF1.0",
    }

    checkpoints = {
        k: os.path.join(model_dir, v, model_spec, "cross_val")
        for k, v in checkpoints.items()
    }
    df = (
        prediction_performance_comparison_table(
            checkpoints=checkpoints,
            metrics=["accuracy", "recall"],
            orientation="h",
        )
        * 100
    )
    print(df)
    with pd.option_context("max_colwidth", 1000):
        print(df.to_latex())
