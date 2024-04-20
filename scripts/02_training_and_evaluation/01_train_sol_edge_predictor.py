""" Training script for solution edge prediction model (supervised with binary labels)."""

from datetime import datetime
from functools import partial
import logging
import os

import hydra
import numpy as np
from omegaconf import DictConfig
from omegaconf import open_dict
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import torch

from core.ml_models.fctp_sol_predictor import GCNNSolEdgePredictor
from core.ml_models.fctp_sol_predictor import (
    EdgeLogRegSolEdgePredictor,
)
from core.ml_models.fctp_sol_predictor import (
    EdgeMLPSolEdgePredictor,
)
from core.utils.ml_utils import FCTPData
from core.utils.ml_utils import StandardNormalizer
from core.utils.ml_utils import MultiInputNormalizer
from core.utils.ml_utils import get_bipartite_raw_features
from core.utils.ml_utils import get_edge_features
from core.utils.ml_utils import get_bipartite_advanced_edge_features
from core.utils.ml_utils import get_raw_targets
from core.utils.ml_utils import get_sample_paths
from core.utils.ml_utils import get_class_weights
from core.utils.ml_utils import training_wrapper
from core.utils.ml_utils import binary_classification_eval_display_func
from core.utils.ml_utils import multiclass_classification_eval_display_func
from core.evaluation.ml_model_evaluation import select_best_model_across_candidates


@hydra.main(
    version_base=None,
    config_path=os.path.join(os.getcwd(), "configs", "training"),
    config_name="config",
)
def main(training_config: DictConfig) -> None:

    # Set random seeds
    seed = training_config.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Extract model config
    model_config = training_config.model
    model_spec = "_".join([f"{k}_{v}" for k, v in model_config.items()])

    # Prepare output directory
    out_dir = os.path.join(training_config.out_dir, model_spec)
    if training_config.cross_validate:
        out_dir = os.path.join(out_dir, "cross_val")
    else:
        out_dir = os.path.join(out_dir, "application")
    os.makedirs(out_dir, exist_ok=True)

    # Set up logger
    os.makedirs(training_config.log_dir, exist_ok=True)
    logger = logging.getLogger()
    logger.addHandler(
        logging.FileHandler(
            os.path.join(
                training_config.log_dir,
                f"{datetime.now().strftime('%Y%m%d_%H:%M:%S')}_train.log",
            ),
            mode="w",
        )
    )

    logger.info(f"Training parameters: {training_config}")

    # Get list of (subset of) sample file paths
    sample_paths = get_sample_paths(
        training_config.data_path, training_config.num_samples
    )

    #######################################################
    # Extract features
    #######################################################

    if model_config.features == "bipartite_raw":
        x_supply, x_demand, x_connections = get_bipartite_raw_features(sample_paths)
        with open_dict(model_config):
            model_config.node_dim = x_supply[0].shape[-1]
            model_config.edge_dim = x_connections[0].shape[-1]
        x = (x_supply, x_demand, x_connections)
    elif model_config.features in [
        "combined_raw_edge_features",
        "advanced_edge_features",
        "advanced_edge_features_plus_stat_features",
    ]:
        x_connections = get_edge_features(
            sample_paths,
            features=model_config.features,
        )
        with open_dict(model_config):
            model_config.edge_dim = x_connections[0].shape[-1]
        x = (x_connections,)
    elif model_config.features in [
        "bipartite_advanced_edge_features",
        "bipartite_advanced_edge_features_plus_stat_features",
    ]:
        x_supply, x_demand, x_connections = get_bipartite_advanced_edge_features(
            sample_paths,
            features=model_config.features,
        )
        with open_dict(model_config):
            model_config.node_dim = x_supply[0].shape[-1]
            model_config.edge_dim = x_connections[0].shape[-1]
        x = (x_supply, x_demand, x_connections)

    #######################################################
    # Extract labels
    #######################################################

    if model_config.prediction_task == "binary_classification":
        y = get_raw_targets(
            sample_paths,
            binary_target=True,
            output_dim=True,
        )
        with open_dict(model_config):
            model_config.edge_output_dim = 1
    else:
        raise ValueError

    #######################################################
    # Define normalization
    #######################################################

    input_transformer = None
    if model_config.normalization == "standard":
        normalizers = tuple([StandardNormalizer() for _ in range(len(x))])
        input_transformer = MultiInputNormalizer(normalizers)

    #######################################################
    # Define policy
    #######################################################

    if model_config.model == "gcnn":
        policy_fun = GCNNSolEdgePredictor
    elif model_config.model == "linear_logreg":
        policy_fun = EdgeLogRegSolEdgePredictor
    elif model_config.model == "edge_mlp":
        policy_fun = EdgeMLPSolEdgePredictor
    else:
        raise ValueError

    #######################################################
    # Define class weighting
    #######################################################

    class_weight_fun = lambda x: None
    if model_config.prediction_task == "binary_classification":
        class_weight_fun = partial(get_class_weights, binary=True)

    #######################################################
    # Define optimizer
    #######################################################

    adam_params = {
        "lr": training_config.learning_rate,
        "betas": (training_config.momentum, 0.999),
        "weight_decay": training_config.weight_decay,
    }
    if training_config.lr_decay:
        lr_schedule = {
            "factor": training_config.lr_decay_factor,
            "patience": training_config.patience,
            "threshold": training_config.opt_threshold,
        }
    else:
        lr_schedule = None

    #######################################################
    # Configure training performance logging
    #######################################################

    initial_eval_display_func = None
    running_eval_display_func = None
    if model_config.prediction_task == "binary_classification":
        initial_eval_display_func = partial(
            binary_classification_eval_display_func, include_train=True
        )
        running_eval_display_func = partial(
            binary_classification_eval_display_func, include_train=False
        )
    elif model_config.prediction_task in ["threeclass_classification"]:
        initial_eval_display_func = partial(
            multiclass_classification_eval_display_func, include_train=True
        )
        running_eval_display_func = partial(
            multiclass_classification_eval_display_func, include_train=False
        )

    #######################################################
    # Training
    #######################################################

    if not training_config.cross_validate:

        # randomly split data set into training and validation data
        split = train_test_split(
            *x, y, test_size=training_config.test_split, random_state=0
        )
        train_data = [split[i] for i in range(len(split)) if i % 2 == 0]
        val_data = [split[i] for i in range(len(split)) if i % 2 == 1]
        x_train, y_train = train_data[:-1], train_data[-1]
        x_val, y_val = val_data[:-1], val_data[-1]

        train_dataset = FCTPData(x_train, y_train)
        val_dataset = FCTPData(x_val, y_val)

        if input_transformer is not None:
            input_transformer.fit(x_train)

        policy = policy_fun(
            model_config=model_config,
            adam_params=adam_params,
            lr_schedule=lr_schedule,
            input_transformer=input_transformer,
            class_weight=class_weight_fun(y_train),
        )

        training_wrapper(
            training_config,
            model_config,
            train_dataset,
            val_dataset,
            policy,
            out_dir,
            logger,
            eval_train_data=False,
            initial_eval_display_func=initial_eval_display_func,
            running_eval_display_func=running_eval_display_func,
        )

    else:
        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        kf.get_n_splits(y)
        for i, (train_index, val_index) in enumerate(kf.split(y)):
            logger.info(f"Training on fold {i}")
            fold_out_dir = os.path.join(out_dir, f"fold_{i}")
            os.makedirs(fold_out_dir, exist_ok=True)

            x_train, y_train = tuple([x_i[train_index] for x_i in x]), y[train_index]
            x_val, y_val = tuple([x_i[val_index] for x_i in x]), y[val_index]

            train_dataset = FCTPData(x_train, y_train)
            val_dataset = FCTPData(x_val, y_val)

            if input_transformer is not None:
                input_transformer.fit(x_train)

            policy = policy_fun(
                model_config=model_config,
                adam_params=adam_params,
                lr_schedule=lr_schedule,
                input_transformer=input_transformer,
                class_weight=class_weight_fun(y_train),
            )

            training_wrapper(
                training_config,
                model_config,
                train_dataset,
                val_dataset,
                policy,
                fold_out_dir,
                logger,
                eval_train_data=False,
                initial_eval_display_func=initial_eval_display_func,
                running_eval_display_func=running_eval_display_func,
            )
        logger.info("Select best model across folds.")
        select_best_model_across_candidates(out_dir)


if __name__ == "__main__":
    main()
