""" Evaluate, compare, and select ML models. """

import os
from shutil import copy

import numpy as np
import pandas as pd
import plotly.express as px
import torch
from torch.utils.data import DataLoader

from core.utils.ml_utils import load_edge_predictor_model
from core.utils.ml_utils import get_bipartite_raw_features
from core.utils.ml_utils import get_edge_features
from core.utils.ml_utils import get_raw_targets
from core.utils.ml_utils import get_sample_paths
from core.utils.ml_utils import FCTPData


##########################################################
# Prediction performance evaluation
##########################################################


def evaluate_prediction_model(
    model,
    data_path,
    num_samples=None,
    features="bipartite_raw",
    prediction_task="binary_classification",
    batch_size=128,
):
    """Evaluate edge prediction model on new samples.

    Parameters
    ----------
    model: BaseSolEdgePredictor
        Edge prediction model.
    data_path: str
        Path to directory with samples.
    features: str, optional
        Type of features to be used.
    prediction_task: str, optional
        Type of prediction task that is assumed.
    num_samples: int, optional
        Maximum number of samples to be included.
    batch_size: int, optional
        Evaluation batch size. Default is 128.

    Returns
    -------
    p: dict
        Model performance dict.

    """
    # Load and prepare data
    sample_paths = get_sample_paths(data_path, num_samples)

    # Features
    if features == "bipartite_raw":
        x_supply, x_demand, x_connections = get_bipartite_raw_features(
            sample_paths,
        )
        x = (x_supply, x_demand, x_connections)
    elif features in [
        "combined_raw_edge_features",
        "advanced_edge_features",
        "advanced_edge_features_plus_stat_features",
    ]:
        x_connections = get_edge_features(sample_paths, features=features)
        x = (x_connections,)
    else:
        raise ValueError

    # Labels
    if prediction_task == "binary_classification":
        y = get_raw_targets(
            sample_paths,
            binary_target=True,
            output_dim=True,
        )
    else:
        raise ValueError

    dataset = FCTPData(x, y)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

    # Evaluate model
    p = model.evaluate({"eval": data_loader})
    return p


##########################################################
# Model selection
##########################################################


def select_best_model_across_candidates(candidates_dir, target_dir=None, metric="loss"):
    """Select best model across multiple candidates and save it separately.

    Parameters
    ----------
    candidates_dir: str
        Directory containing candidate checkpoints.
    target_dir: str, optional
        Directory to save selected model. If not provided, the source directory is used.
    metric: str, optional
        Metric for model selection ('accuracy', 'recall', 'loss'). Default is validation loss.

    Returns
    -------
    None

    """
    # Set up target directory
    if target_dir is None:
        target_dir = candidates_dir
    os.makedirs(target_dir, exist_ok=True)

    # Identify model directories corresponding to different candidates
    candidate_dirs = [
        d
        for d in os.listdir(candidates_dir)
        if os.path.isdir(os.path.join(candidates_dir, d))
    ]  # and re.match("[0-9]+", d)

    # Define comparison function depending on KPI
    assert metric in ["accuracy", "recall", "loss"]
    if metric == "loss":
        is_better = lambda x, y: x < y
        best_value = float("inf")
    else:
        is_better = lambda x, y: x > y
        best_value = 0

    # Identify best model
    best_chkpnt = None
    for candidate in candidate_dirs:
        chkpnt_path = os.path.join(candidates_dir, candidate, "best_checkpoint.pth.tar")
        chkpnt = torch.load(chkpnt_path, map_location="cpu")
        value = chkpnt["exp_dict"]["performance"][f"validation_{metric}"][-1]
        if is_better(value, best_value):
            best_chkpnt = chkpnt_path
            best_value = value

    # Copy best model to target directory
    copy(best_chkpnt, os.path.join(target_dir, "best_checkpoint.pth.tar"))


##########################################################
# ML model performance plots and tables
##########################################################


def performance_over_epochs_comparison(
    checkpoints,
    metric="loss",
    title=None,
    legend_title=None,
):
    """Plot performance over epochs for different models for comparison.

    Parameters
    ----------
    checkpoints: dict
        Dictionary {<name>: <path>} containing checkpoints to be considered.
    metric: str, optional
        Metric to be plotted.
    title: str, optional
        Plot title.
    legend_title: str, optional
        Legend title.

    Returns
    -------
    None

    """

    # load checkpoints and retrieve performance arrays
    data = {}
    for config, chkpnt_path in checkpoints.items():
        chkpnt = torch.load(chkpnt_path, map_location="cpu")
        data[config] = chkpnt["exp_dict"]["performance"][f"validation_{metric}"]
    max_n = max([len(i) for i in data.values()])
    for config, values in data.items():
        pad_n = max_n - len(values)
        data[config] = values + [None for _ in range(pad_n)]
    data = pd.DataFrame(data)
    data = data.sort_index(axis=1)

    # create a line plot to plot performance against epoch
    fig = px.line(data, markers=False, template="simple_white")
    fig.update_layout(
        yaxis_title=metric.capitalize(),
        xaxis_title="Training epoch",
        legend_title=legend_title,
        title=title,
        legend=dict(yanchor="top", y=1, xanchor="right", x=1),
        font_size=14,
    )
    return fig


def prediction_performance_comparison_table(
    checkpoints,
    cross_val=True,
    metrics=None,
    rounding=None,
    orientation="v",
):
    """Generate prediction performance comparison table.

    Parameters
    ----------
    checkpoints: dict
        Dictionary {<name>: <path>} containing checkpoints to be considered.
    cross_val: bool, optional
        Indicate whether performance has to be aggregated over multiple folds (mean). Default is
        True.
    metrics: list, optional
        Metrics to be included.
    rounding: dict, optional
        Rounding to be applied to the different metrics.
    orientation: str, optional
        Table orientation. Can be horizontal ('h') or vertical ('v'). Default is 'v'.

    Returns
    -------
    None

    """

    if metrics is None:
        metrics = ["loss", "accuracy", "recall"]
    if rounding is None:
        rounding = {"loss": 3, "accuracy": 4, "recall": 4}

    # load checkpoints and retrieve performance arrays
    data = pd.DataFrame(index=checkpoints.keys(), columns=metrics)
    for config, chkpnt_path in checkpoints.items():
        if not cross_val:
            chkpnt = torch.load(chkpnt_path, map_location="cpu")
            for metric in metrics:
                # average over last five epochs to smoothen
                data.loc[config][metric] = np.mean(
                    chkpnt["exp_dict"]["performance"][f"validation_{metric}"][-5:]
                )
        else:
            fold_performance = {k: [] for k in metrics}
            folds = [
                p
                for p in os.listdir(chkpnt_path)
                if os.path.isdir(os.path.join(chkpnt_path, p))
            ]
            for fold in folds:
                chkpnt_fold_path = os.path.join(
                    chkpnt_path, fold, "best_checkpoint.pth.tar"
                )
                chkpnt = torch.load(chkpnt_fold_path, map_location="cpu")
                for metric in metrics:
                    fold_performance[metric] = np.mean(
                        chkpnt["exp_dict"]["performance"][f"validation_{metric}"][-5:]
                    )
            for metric in metrics:
                data.loc[config][metric] = np.mean(fold_performance[metric])

    # display table in Latex format and save as csv
    data = data.apply(pd.to_numeric).round(rounding)
    if orientation == "h":
        data = data.transpose()
    return data


def prediction_performance_comparison_boxplot(
    checkpoints,
    metric="loss",
    smoothing=True,
    yaxis_title=None,
    xaxis_title=None,
    title=None,
):
    """Compare different models via performance boxplots.

    Parameters
    ----------
    checkpoints: dict
        Dictionary {<name>: <path>} containing checkpoints to be considered.
    metric: str, optional
        Metric to be used for comparison.
    smoothing: bool, optional
        Indicate whether final value or average over last five epochs
        (smoothing=True) should be used.
    yaxis_title: str, optional
        Title for y-axis. If not provided, <metric> will be used.
    xaxis_title: str, optional
        Title for x-axis.
    title: str, optional
        Plot title.

    Returns
    -------
    None

    """

    # load checkpoints and retrieve performance arrays
    folds = list(
        {
            p
            for chkpnt_path in checkpoints.values()
            for p in os.listdir(chkpnt_path)
            if os.path.isdir(os.path.join(chkpnt_path, p))
        }
    )
    data = pd.DataFrame(columns=checkpoints.keys(), index=folds)
    for config, chkpnt_path in checkpoints.items():
        for fold in folds:
            chkpnt_fold_path = os.path.join(
                chkpnt_path, fold, "best_checkpoint.pth.tar"
            )
            chkpnt = torch.load(chkpnt_fold_path, map_location="cpu")
            if smoothing:
                perf = np.mean(
                    chkpnt["exp_dict"]["performance"][f"validation_{metric}"][-5:]
                )
            else:
                perf = chkpnt["exp_dict"]["performance"][f"validation_{metric}"][-1]
            data.loc[fold][config] = perf

    # display as boxplot
    fig = px.box(data, y=list(checkpoints.keys()), template="simple_white")
    fig.update_yaxes(range=get_y_range(data, additional_values=[], offset=0.1))
    fig.update_layout(
        yaxis_title=metric if yaxis_title is None else yaxis_title,
        xaxis_title=xaxis_title,
        title=title,
        font_size=18,
    )
    return fig


def get_y_range(data, additional_values=None, column=None, offset=0.1):
    """Calculate y-axis range based on values and specified offset.

    Find minimum and maximum values and add some offset.

    Parameters
    ----------
    data: pd.DataFrame
        Data provided as Dataframe.
    baseline_values: list, optional
        Additional values to be considered.
    column: str, optional
        Specific column to use for determining min and max values.
    offset: float, optional
        Relative offset. Default is 10%.

    Returns
    -------
    list
        Minimum and maximum y-value.

    """
    if additional_values is None:
        additional_values = []
    if column is None:
        min_y = min([data.min().min(), *additional_values])
        max_y = max([data.max().max(), *additional_values])
    else:
        min_y = min([data[column].min(), *additional_values])
        max_y = max([data[column].max(), *additional_values])
    delta = max_y - min_y
    min_y -= offset * delta
    max_y += offset * delta
    return [min_y, max_y]


def generalization_performance_comparison_across_datasets(
    transfer_combinations,
    cross_val=True,
    metrics=None,
    rounding=None,
):
    """Evaluate generalization performance by evaluating different combinations of models and
    datasets.

    Parameters
    ----------
    transfer_combinations: dict
        Transfer combinations to be considered: {<name>: (<chkpnt_path>, <data_path>)}.
    cross_val: bool, optional
        Indicate whether performance has to be aggregated over multiple folds (mean). Default is
        True.
    metrics: list, optional
        Metrics to be included.
    rounding: dict, optional
        Rounding to be applied to the different metrics.

    Returns
    -------
    data: pd.DataFrame
        Performance table as dataframe.

    """
    if metrics is None:
        metrics = ["loss", "accuracy", "recall"]
    if rounding is None:
        rounding = {"loss": 3, "accuracy": 4, "recall": 4}

    data = pd.DataFrame(index=transfer_combinations.keys(), columns=metrics)

    for config, (chkpnt_path, transfer_data) in transfer_combinations.items():
        try:
            if not cross_val:
                model = load_edge_predictor_model(
                    chkpnt_path, get_feature_fun=False, get_label_fun=False
                )
                model_config = torch.load(chkpnt_path, map_location="cpu")[
                    "model_config"
                ]
                p = evaluate_prediction_model(
                    model,
                    transfer_data,
                    features=model_config.features,
                    prediction_task=model_config.prediction_task,
                )
                for metric in metrics:
                    data.loc[config][metric] = p[f"eval_{metric}"]
            else:
                fold_performance = {k: [] for k in metrics}
                folds = [
                    p
                    for p in os.listdir(chkpnt_path)
                    if os.path.isdir(os.path.join(chkpnt_path, p))
                ]
                for fold in folds:
                    chkpnt_fold_path = os.path.join(
                        chkpnt_path, fold, "best_checkpoint.pth.tar"
                    )
                    model = load_edge_predictor_model(
                        chkpnt_fold_path, get_feature_fun=False, get_label_fun=False
                    )
                    model_config = torch.load(chkpnt_fold_path, map_location="cpu")[
                        "model_config"
                    ]
                    p = evaluate_prediction_model(
                        model,
                        transfer_data,
                        features=model_config.features,
                        prediction_task=model_config.prediction_task,
                    )
                    for metric in metrics:
                        fold_performance[metric] = p[f"eval_{metric}"]
                for metric in metrics:
                    data.loc[config][metric] = np.mean(fold_performance[metric])
        except:
            pass

    data = data.apply(pd.to_numeric).round(rounding)

    return data
