""" Evaluate and report generalization performance of GNN models (Table 5).

Following transfer combinations require temporary adjustments of feature extraction functions in
core.utils.ml_utils:

FCTP -> C-FCTP:
Ignore edge capacity feature, i.e., treat as FCTP
>>> x_e = np.dstack((instance.var_costs, instance.fix_costs))

FCTP -> SF-FCTP:
Disregrad fixed-step cost structure, i.e., calculate approx. fix costs and treat as FCTP
.. math:: f_ij' = ceil(min{s_i, d_j}/cap_ij) * f_ij
>>> from core.utils.preprocessing import min_of_supply_demand
>>> fix_costs_prime = (
        np.ceil(
            min_of_supply_demand(instance.supply, instance.demand)
            / instance.vehicle_capacities
        )
        * instance.fix_costs
    ).astype(int)
>>> x_e = np.dstack((instance.var_costs, fix_costs_prime))

"""

import os

import pandas as pd

from core.evaluation.ml_model_evaluation import (
    generalization_performance_comparison_across_datasets,
)


if __name__ == "__main__":

    model_dir = "trained_models/sol_edge_predictor"
    model_spec = "gcnn/model_gcnn_features_bipartite_raw_prediction_task_binary_classification_normalization_standard_hidden_layer_dim_20_num_conv_layers_10_num_dense_layers_2"
    sample_dir = "data_paper/samples"

    transfer_combinations = {
        "value range 50 -> 20": (
            "fctp_agarwal-aneja_15_15_B50_Theta0.2_BF1.0",
            "fctp_agarwal-aneja_15_15_B20_Theta0.2_BF1.0",
        ),
        "value range 20 -> 50": (
            "fctp_agarwal-aneja_15_15_B20_Theta0.2_BF1.0",
            "fctp_agarwal-aneja_15_15_B50_Theta0.2_BF1.0",
        ),
        "supply surplus 0% -> 30%": (
            "fctp_agarwal-aneja_15_15_B20_Theta0.2_BF1.0",
            "fctp_agarwal-aneja_15_15_B20_Theta0.2_BF1.3",
        ),
        "supply surplus 30% -> 0%": (
            "fctp_agarwal-aneja_15_15_B20_Theta0.2_BF1.3",
            "fctp_agarwal-aneja_15_15_B20_Theta0.2_BF1.0",
        ),
        "agarwal-aneja -> euclidian": (
            "fctp_agarwal-aneja_15_15_B20_Theta0.2_BF1.0",
            "fctp_euclidian_15_15_B20_Theta0.2_BF1.0",
        ),
        "euclidian -> agarwal-aneja": (
            "fctp_euclidian_15_15_B20_Theta0.2_BF1.0",
            "fctp_agarwal-aneja_15_15_B20_Theta0.2_BF1.0",
        ),
        "flow cost contrib 0.5 -> 0.0": (
            "fctp_agarwal-aneja_15_15_B20_Theta0.5_BF1.0",
            "fctp_agarwal-aneja_15_15_B20_Theta0.0_BF1.0",
        ),
        "flow cost contrib 0.0 -> 0.5": (
            "fctp_agarwal-aneja_15_15_B20_Theta0.0_BF1.0",
            "fctp_agarwal-aneja_15_15_B20_Theta0.5_BF1.0",
        ),
        "size 15x15 -> 30x30": (
            "fctp_agarwal-aneja_15_15_B20_Theta0.2_BF1.0",
            "fctp_agarwal-aneja_30_30_B20_Theta0.2_BF1.0",
        ),
        # "FCTP -> C-FCTP": (
        #     "fctp_agarwal-aneja_15_15_B20_Theta0.2_BF1.0",
        #     "c-fctp_agarwal-aneja_15_15_B20_Theta0.2_BF1.0",
        # )
        # "FCTP -> FS-FCTP": (
        #     "fctp_agarwal-aneja_15_15_B20_Theta0.2_BF1.0",
        #     "fs-fctp_agarwal-aneja_15_15_B20_Theta0.2_BF1.0",
        # )
    }

    transfer_combinations = {
        k: (
            os.path.join(model_dir, m, model_spec, "cross_val"),
            os.path.join(sample_dir, d),
        )
        for k, (m, d) in transfer_combinations.items()
    }

    df = (
        generalization_performance_comparison_across_datasets(
            transfer_combinations,
            cross_val=True,
            metrics=["accuracy", "recall"],
        )
        * 100
    )
    print(df)
    with pd.option_context("max_colwidth", 1000):
        print(df.to_latex())
