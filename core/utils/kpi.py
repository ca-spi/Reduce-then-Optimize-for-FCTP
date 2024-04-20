""" Functions for evaluating FCTP instances and results. """

import numpy as np
import torch


def varcoeff(data):
    """Compute coefficient of variation for a vector of values.

    Parameters
    ----------
    data: list of float
        Data for which coefficient of variation should be computed.

    Returns
    -------
    float
        Coefficient of variation.

    """
    return np.std(data) / np.mean(data)


def stats(data):
    """Compute some basic statistics for a vector of values.

    Parameters
    ----------
    data: list of float
        Data for which statistics should be computed.

    Returns
    -------
    dict
        Dictionary of statistics describing distribution of data vector.

    """
    return {
        "mean": np.mean(data),
        "median": np.median(data),
        "min": np.min(data),
        "max": np.max(data),
        "5%-quantile": np.quantile(data, 0.05),
        "95%-quantile": np.quantile(data, 0.95),
        "std": np.std(data),
    }


def gmean(data):
    """Geometric mean.

    Parameters
    ----------
    data: list of float
        Data for which mean should be computed.

    Returns
    -------
    float
        Geometric mean.

    """
    data = np.log(data)
    return np.exp(data.mean())


def shifted_gmean(data, s=0):
    """Shifted geometric mean, following the definition of
    https://www.math.uwaterloo.ca/~bico/papers/exactmip_bb_short.pdf

    Typical values:
    - time: s=10

    Parameters
    ----------
    data: list of float
        Data for which mean should be computed.
    s: int, optional
        Shift. Default is 0.

    Returns
    -------
    float
        Shifted geometric mean.

    """
    data = np.array(data) + s
    return gmean(data) - s


def evaluate_solution_costs_dict(solution, var_costs, fix_costs):
    """Compute total costs for a FCTP solution provided as dict.

    Parameters
    ----------
    solution: dict
        Solution dictionary containing flows from supplier i to customer j.

    var_costs: 2D np.array
        Variables costs for sending one unit of flow from supplier i to customer j.

    fix_costs: 2D np.array
        Fixed costs for sending flow from supplier i to customer j.

    Returns
    -------
    float or int
        Total costs incurred by solution.

    """
    v_costs = sum([v * var_costs[i, j] for (i, j), v in solution.items()])
    f_costs = sum([fix_costs[i, j] for (i, j), v in solution.items() if v > 0])
    return v_costs + f_costs


def evaluate_solution_costs_matrix(solution_matrix, cost_matrix, batch=True):
    """Compute total costs for a FCTP solution provided as matrix.

    Parameters
    ----------
    solution: 2D np.array
        Solution matrix containing flows from supplier i to customer j.

    cost_matrix: 3D np.array
        Variable and fixed costs for sending flow from supplier i to customer j.

    Returns
    -------
    float or int
        Total costs incurred by solution.

    """
    if batch:
        var_costs = cost_matrix[:, :, :, 0]
        fix_costs = cost_matrix[:, :, :, 1]
        total_costs = np.sum(
            solution_matrix * var_costs
            + np.where(solution_matrix > 0, 1, 0) * fix_costs,
            axis=(1, 2),
        )
    else:
        var_costs = cost_matrix[:, :, 0]
        fix_costs = cost_matrix[:, :, 1]
        total_costs = np.sum(
            solution_matrix * var_costs
            + np.where(solution_matrix > 0, 1, 0) * fix_costs,
        )
    return total_costs


def eval_edge_prediction_accuracy(pred, y):
    """Get prediction accuracy metrics for binary classification.

    Metrics: accuracy, recall, precision, f-score.

    Parameters
    ----------
    pred: array-like
        Predictions (0 or 1).
    y: array-like
        Targets (0 or 1).

    Returns
    -------
    tuple
        Accuracy, recall, precision, and f-score.

    """
    assert y.max() <= 1 and pred.max() <= 1, "targets and predictions need to be binary"
    return (
        get_accuracy(pred, y),
        get_recall(pred, y),
        get_precision(pred, y),
        get_f_score(pred, y),
    )


def get_recall(prediction, target):
    """Calculate recall.

    Parameters
    ----------
    prediction: array-like
        Predictions (0 or 1).
    target: array-like
        Targets (0 or 1).

    Returns
    -------
    float
        Recall.

    """
    return 1 - np.sum(np.where(target > prediction, 1, 0)) / np.sum(target)


def get_precision(prediction, target):
    """Calculate precision.

    Parameters
    ----------
    prediction: array-like
        Predictions (0 or 1).
    target: array-like
        Targets (0 or 1).

    Returns
    -------
    float
        Precision.

    """
    return np.sum((prediction == target) * target) / np.sum(prediction)


def get_accuracy(prediction, target):
    """Calculate accuracy.

    Parameters
    ----------
    prediction: array-like
        Predictions (0 or 1).
    target: array-like
        Targets (0 or 1).

    Returns
    -------
    float
        Accuracy.

    """
    return np.mean(prediction == target)


def f_score(precision, recall, beta=2):
    """Calculate F-score from precision and recall.

    Parameters
    ----------
    precision: float
        Precision.
    recall: float
        Recall.

    Returns
    -------
    float
        F-score.

    """
    return (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)


def get_f_score(prediction, target, beta=2):
    """Calculate F-score from predictions and targets.

    Parameters
    ----------
    prediction: array-like
        Predictions (0 or 1).
    target: array-like
        Targets (0 or 1).

    Returns
    -------
    float
        F-score.

    """
    precision = get_precision(prediction, target)
    recall = get_recall(prediction, target)
    return f_score(precision, recall, beta)


def eval_model_performance(
    model, x_train, y_train, x_validation, y_validation, loss_func
):
    """Evaluate performance of a binary classification model (pytorch) on training and test data.

    Parameters
    ----------
    model: pytorch model
        Binary classification model.

    x_train: np.array
        Training input data.

    y_train: np.array
        Training labels.

    x_validation: np.array
        Validation input data.

    y_validation: np.array
        Validation labels.

    loss_func: func
        Loss function that should be evaluated

    Returns
    -------
    p: dict
        Dictionary containing different performance KPIs.
    """
    model.eval()

    ##### Performance on training data

    train_outputs = model(*x_train)

    # evaluate loss
    train_loss = loss_func(prediction=train_outputs, label=y_train)

    # evaluate prediction accuracy (correct predictions) and recall (correctly predicted positives)
    labels_train = y_train.detach().numpy()
    predictions_train = np.round(torch.sigmoid(train_outputs).detach().numpy())
    accuracy_train = np.sum((labels_train == predictions_train)) / np.prod(
        labels_train.shape
    )
    recall_train = 1 - np.sum(
        np.where(labels_train > predictions_train, 1, 0)
    ) / np.sum(labels_train)

    ##### Performance on validation data

    validation_outputs = model(*x_validation)

    # evaluate loss
    validation_loss = loss_func(prediction=validation_outputs, label=y_validation)

    # evaluate prediction accuracy (correct predictions) and recall (correctly predicted positives)
    labels_validation = y_validation.detach().numpy()
    predictions_validation = np.round(
        torch.sigmoid(validation_outputs).detach().numpy()
    )
    accuracy_validation = np.sum(
        (labels_validation == predictions_validation)
    ) / np.prod(labels_validation.shape)
    recall_validation = 1 - np.sum(
        np.where(labels_validation > predictions_validation, 1, 0)
    ) / np.sum(labels_validation)

    p = {
        "Train_Loss": train_loss.item(),
        "Train_Accuracy": accuracy_train,
        "Train_Recall": recall_train,
        "Validation_Loss": validation_loss.item(),
        "Validation_Accuracy": accuracy_validation,
        "Validation_Recall": recall_validation,
    }

    model.train()

    return p
