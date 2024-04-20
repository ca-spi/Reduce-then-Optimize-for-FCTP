""" Generate hyperparameter screeening plots (Figure 13). """

import glob
import os
import re

from core.evaluation.ml_model_evaluation import performance_over_epochs_comparison
from core.utils.visualization import warmstart_px_save

if __name__ == "__main__":

    model_dir = "trained_models/sol_edge_predictor/fctp_agarwal-aneja_15_15_B20_Theta0.2_BF1.0/gcnn/hyperparam_screening"
    metric = "loss"
    out_dir = "results/hyperparam_screening"

    defaults = {"nconv": 10, "ndense": 2, "dim": 20}

    # Retrieve relevant checkpoints
    for param in defaults.keys():
        print(f"Evaluating parameter {param}...")
        if param == "nconv":
            pattern = os.path.join(
                model_dir,
                f"*dim_{defaults['dim']}_num_conv_layers_*_num_dense_layers_{defaults['ndense']}*",
            )
            get_param_val = lambda n: int(
                re.findall(r"num_conv_layers_[0-9]+", n)[0].split("num_conv_layers_")[1]
            )
            legend_title = "#Convolutional layers"
        elif param == "ndense":
            pattern = os.path.join(
                model_dir,
                f"*dim_{defaults['dim']}_num_conv_layers_{defaults['nconv']}_num_dense_layers_*",
            )
            get_param_val = lambda n: int(
                re.findall(r"num_dense_layers_[0-9]+", n)[0].split("num_dense_layers_")[
                    1
                ]
            )
            legend_title = "#Dense layers"
        else:
            pattern = os.path.join(
                model_dir,
                f"*dim_*_num_conv_layers_{defaults['nconv']}_num_dense_layers_{defaults['ndense']}*",
            )
            get_param_val = lambda n: int(
                re.findall(r"dim_[0-9]+", n)[0].split("dim_")[1]
            )
            legend_title = "Feature dimension"
        relevant_models = glob.glob(pattern)
        relevant_models = {
            get_param_val(os.path.basename(m)): os.path.join(
                m, "application", "checkpoint.pth.tar"
            )
            for m in relevant_models
        }

        # Generate plot
        fig = performance_over_epochs_comparison(
            checkpoints=relevant_models,
            metric=metric,
            title=None,
            legend_title=legend_title,
        )

        # Save plot
        os.makedirs(out_dir, exist_ok=True)
        filename = f"hyperparam_screening_{param}_{metric}.pdf"
        img_path = os.path.join(out_dir, filename)
        warmstart_px_save(img_path)
        fig.write_image(img_path)
