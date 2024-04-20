""" Report benchmarking performance (runtimes, opt gaps) for different FCTP types (Table 7). """

from collections import OrderedDict
import os

import pandas as pd

from core.evaluation.benchmarking_utils import get_summary_performance_table


if __name__ == "__main__":

    benchmarking_dir = "benchmarking_paper"

    relevant_keys = OrderedDict(
        {
            "opt gap": "mean",
            "runtime": "average solver runtime",
        }
    )
    relevant_edgeset_sizes = [0.4, 0.3, 0.2]
    baseline = "exact/grb_timeout_43200"
    model_spec = "model_gcnn_features_bipartite_raw_prediction_task_binary_classification_normalization_standard_hidden_layer_dim_20_num_conv_layers_10_num_dense_layers_2"
    relevant_methods = OrderedDict(
        {
            s: f"ml-reduction/size/exact-grb_timeout_43200/{model_spec}/{s}"
            for s in relevant_edgeset_sizes
        }
    )
    relevant_methods["exact"] = baseline
    relevant_methods.move_to_end("exact", last=False)

    solution_dirs = {
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

    solution_dirs = {
        k: os.path.join(benchmarking_dir, v) for k, v in solution_dirs.items()
    }
    summary_df_tmp = get_summary_performance_table(
        solution_dirs,
        relevant_keys=relevant_keys,
        relevant_methods=relevant_methods.values(),
        baseline=baseline,
        use_gmean=True,
    )
    column_index = pd.MultiIndex.from_product(
        [relevant_methods, relevant_keys], names=["method", "KPI"]
    )
    summary_df = pd.DataFrame(index=solution_dirs.keys(), columns=column_index)
    for dataset in solution_dirs.keys():
        for method_new, method_old in relevant_methods.items():
            for kpi in relevant_keys:
                summary_df.loc[dataset][(method_new, kpi)] = summary_df_tmp[dataset][
                    kpi
                ][method_old]
    summary_df = summary_df.drop(columns=[("exact", "opt gap")])

    print(summary_df)
    with pd.option_context("max_colwidth", 1000):
        print(summary_df.to_latex())
