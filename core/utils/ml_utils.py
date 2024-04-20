""" Utility functions for training and evaluating ML models. """

from collections import defaultdict
import datetime
from functools import partial
import os
import random
import time

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from core.data_processing.data_utils import load_sample
from core.utils.fctp import CapacitatedFCTP
from core.utils.fctp import FixedStepFCTP
from core.fctp_solvers.heuristics import north_west_corner
from core.ml_models.fctp_sol_predictor import GCNNSolEdgePredictor
from core.ml_models.fctp_sol_predictor import EdgeLogRegSolEdgePredictor
from core.ml_models.fctp_sol_predictor import EdgeMLPSolEdgePredictor
from core.utils.kpi import evaluate_solution_costs_dict
from core.utils.postprocessing import sol_to_matrix
from core.utils.utils import default_to_regular_dict

###################################
# Data processing
###################################


def get_sample_paths(sample_dir, num_samples=None):
    """Get list of sample paths.

    Parameters
    ----------
    sample_dir: str
        Directory containing sample files.
    num_samples: int, optional
        Maximum number of samples.

    Returns
    -------
    sample_paths: list
        List of sample paths.

    """
    sample_paths = [
        os.path.join(sample_dir, filename) for filename in os.listdir(sample_dir)
    ]
    if num_samples is not None:
        np.random.shuffle(sample_paths)
        sample_paths = sample_paths[:num_samples]
    return sample_paths


###################################
# Feature generation
###################################


def get_bipartite_raw_features_for_instance(instance, batch_dim=True):
    """Translate instance into feature arrays for bipartite GNN.

    Features:
    - supply nodes: supply
    - demand nodes: demand
    - edges: variable costs, fixed costs, (capacities)

    Parameters
    ----------
    instance: tuple
        Instance tuple.
    batch_dim: bool, optional
        Indicate whether batch dimension should be added.

    Returns
    -------
    tuple
        3-element tuple containing supply node features, demand node features,
        and edge features.

    """
    x_s = np.array(instance.supply)[:, None]
    x_d = np.array(instance.demand)[:, None]
    if isinstance(instance, CapacitatedFCTP):
        x_e = np.dstack(
            (instance.var_costs, instance.fix_costs, instance.edge_capacities)
        )
    elif isinstance(instance, FixedStepFCTP):
        x_e = np.dstack(
            (instance.var_costs, instance.fix_costs, instance.vehicle_capacities)
        )
    else:
        x_e = np.dstack((instance.var_costs, instance.fix_costs))

    if batch_dim:
        x_s = x_s[None, :]
        x_d = x_d[None, :]
        x_e = x_e[None, :, :, :]

    return (x_s, x_d, x_e)


def get_bipartite_raw_features(
    sample_paths,
):
    """Translate a set of N sample instances into feature arrays for bipartite GNN.

    Features:
    - supply nodes: supply (N x num_suppliers x 1)
    - demand nodes: demand (N x num_customers x 1)
    - edges: variable costs, fixed costs, (capacities) (N x num_suppliers x num_customers x 2 or 3)

    Parameters
    ----------
    sample_paths: list
        List of sample paths.

    Returns
    -------
    tuple
        3-element tuple containing supply node features, demand node features,
        and edge features.

    """

    x_supply = []
    x_demand = []
    x_connection = []

    # Read samples from pickle files
    for sample_path in sample_paths:
        sample = load_sample(sample_path)
        # Prepare features
        x_s, x_d, x_e = get_bipartite_raw_features_for_instance(
            sample["instance"], batch_dim=True
        )
        x_supply.append(x_s)
        x_demand.append(x_d)
        x_connection.append(x_e)

    x_supply = np.concatenate(x_supply)
    x_demand = np.concatenate(x_demand)
    x_connection = np.concatenate(x_connection)

    return x_supply, x_demand, x_connection


def get_bipartite_advanced_edge_features_for_instance(
    instance,
    include_statistical_features=False,
    batch_dim=True,
):
    """Translate instance into feature arrays for bipartite GNN, including advanced edge features.

    Features:
    - supply nodes: supply
    - demand nodes: demand
    - edges: advanced edge features

    Parameters
    ----------
    instance: tuple
        Instance tuple.
    include_statistical_features: bool, optional
        Indicate whether statistical features should be included. Default is
        False.
    batch_dim: bool, optional
        Indicate whether batch dimension should be added.

    Returns
    -------
    tuple
        3-element tuple containing supply node features, demand node features,
        and edge features.

    """
    supply = instance.supply
    demand = instance.demand

    edge_features = get_advanced_edge_features_for_instance(
        instance,
        include_node_features=False,
        include_statistical_features=include_statistical_features,
        batch_dim=batch_dim,
    )

    if batch_dim:
        supply_features = np.array(supply)[None, :, None]
        demand_features = np.array(demand)[None, :, None]
    else:
        supply_features = np.array(supply)[:, None]
        demand_features = np.array(demand)[:, None]
    return supply_features, demand_features, edge_features


def get_bipartite_advanced_edge_features(
    sample_paths,
    features="bipartite_advanced_edge_features",
):
    """Translate a set of N sample instances into feature arrays for bipartite GNN, including
    advanced edge features.

    Features:
    - supply nodes: supply (N x num_suppliers x 1)
    - demand nodes: demand (N x num_customers x 1)
    - edges: advanced edge features (N x num_suppliers x num_customers x num_features)

    Parameters
    ----------
    sample_paths: list
        List of sample paths.
    features: str, optional
        Type of features. Can be 'bipartite_advanced_edge_features' or
        'bipartite_advanced_edge_features_plus_stat_features'.

    Returns
    -------
    tuple
        3-element tuple containing supply node features, demand node features,
        and edge features.

    """
    if features == "bipartite_advanced_edge_features":
        feature_fun = get_bipartite_advanced_edge_features_for_instance
    elif features == "bipartite_advanced_edge_features_plus_stat_features":
        feature_fun = partial(
            get_bipartite_advanced_edge_features_for_instance,
            include_statistical_features=True,
        )

    x_connection = []
    x_supply = []
    x_demand = []
    # Read samples from pickle files
    for sample_path in sample_paths:
        sample = load_sample(sample_path)
        # Prepare features
        x_s, x_d, x_e = feature_fun(sample["instance"], batch_dim=True)
        x_supply.append(x_s)
        x_demand.append(x_d)
        x_connection.append(x_e)

    x_connection = np.concatenate(x_connection)
    x_supply = np.concatenate(x_supply)
    x_demand = np.concatenate(x_demand)

    return x_supply, x_demand, x_connection


def get_combined_raw_edge_features_for_instance(instance, batch_dim=True):
    """Translate instance into combined edge features for baseline models.

    Features:
    - edges: supply, demand, variable costs, fixed costs, (capacities)

    Parameters
    ----------
    instance: tuple
        Instance tuple.
    batch_dim: bool, optional
        Indicate whether batch dimension should be added.

    Returns
    -------
    np.array
        Edge features.

    """
    supply_exp = np.repeat(
        np.expand_dims(instance.supply, axis=1), repeats=instance.n, axis=1
    )
    demand_exp = np.repeat(
        np.expand_dims(instance.demand, axis=0), repeats=instance.m, axis=0
    )
    if isinstance(instance, CapacitatedFCTP):
        edge_features = np.dstack(
            (
                supply_exp,
                demand_exp,
                instance.var_costs,
                instance.fix_costs,
                instance.edge_capacities,
            )
        )
    elif isinstance(instance, FixedStepFCTP):
        edge_features = np.dstack(
            (
                supply_exp,
                demand_exp,
                instance.var_costs,
                instance.fix_costs,
                instance.vehicle_capacities,
            )
        )
    else:
        edge_features = np.dstack(
            (supply_exp, demand_exp, instance.var_costs, instance.fix_costs)
        )

    if batch_dim:
        return edge_features[None, :, :, :]
    else:
        return edge_features


def get_num_random_sols(m, n, exp_occ=100):
    """Calculate how many random solutions are needed to observe edges with a certain frequency.

    Parameters
    ----------
    m: int
        Number of supply nodes.
    n: int
        Number of demand nodes.
    exp_occ: int
        Target frequency.

    Returns
    -------
    int
        Number of random solutions that should be generated.

    """
    return int(exp_occ * ((m * n) / (m + n - 1)))


def get_random_sol(supply, demand):
    """Get random solution.

    Shuffle instance and generate solution by applying North-West-Corner Method.

    Parameters
    ----------
    supply: 1D np.array or list
        A list of supplier capacities/supplies.
    demand: 1D np.array or list
        A list of customer demands.

    Returns
    -------
    sol: dict
        Solution dictionary.

    """
    # Randomly permute instance
    n_supply = len(supply)
    n_demand = len(demand)

    supply_mapping = list(range(n_supply))
    random.shuffle(supply_mapping)

    demand_mapping = list(range(n_demand))
    random.shuffle(demand_mapping)

    supply_shuffled = np.array(
        [supply[supply_mapping[i]] for i in range(n_supply)], dtype=int
    )
    demand_shuffled = np.array(
        [demand[demand_mapping[i]] for i in range(n_demand)], dtype=int
    )

    sol_shuffled = north_west_corner(supply_shuffled, demand_shuffled)
    sol = {
        (supply_mapping[i], demand_mapping[j]): v for (i, j), v in sol_shuffled.items()
    }

    return sol


def get_statistical_edge_features(instance):
    """Get statistical edge features.

    Features:
    - edges: ranking-based score, correlation-based score

    Parameters
    ----------
    instance: tuple
        Instance tuple.

    Returns
    -------
    edge_ranks: np.array
        Matrix containing rank-based score for each edge.
    edge_correlations: np.array
        Matrix containing correlation-based score for each edge.

    """
    supply, demand, var_costs, fix_costs = (
        instance.supply,
        instance.demand,
        instance.var_costs,
        instance.fix_costs,
    )
    # Generate random solutions and evaluate their solution quality
    num_random_sols = get_num_random_sols(len(supply), len(demand))
    sols = []
    for _ in range(num_random_sols):
        sol = get_random_sol(supply, demand)
        sol_val = evaluate_solution_costs_dict(sol, var_costs, fix_costs)
        sols.append((sol_val, sol))

    # Get average solution value
    sol_vals = np.array([sol[0] for sol in sols])
    avg_sol_val = np.mean(sol_vals)
    sol_val_diff = (sol_vals - avg_sol_val).sum()
    sol_val_var = np.var(sol_vals)

    # Sort solutions to obtain ranking
    sols = sorted(sols, key=lambda x: x[0])

    edges = [(i, j) for i in range(len(supply)) for j in range(len(demand))]
    # For each edge, collect the ranks of the solutions it is part of (with positive flow)
    avg_edge_ranks = {e: 0 for e in edges}
    x_bar = {e: 0 for e in edges}
    s1 = {e: 0 for e in edges}
    for i, (sol_val, sol) in enumerate(sols):
        rank = i + 1
        for edge, val in sol.items():
            if val > 0:
                avg_edge_ranks[edge] += 1 / rank
                x_bar[edge] += 1 / num_random_sols
                s1[edge] += sol_val - avg_sol_val

    edge_correlations = {}
    for edge in edges:
        sigma_c = (1 - x_bar[edge]) * s1[edge] - x_bar[edge] * (sol_val_diff - s1[edge])
        sigma_x = x_bar[edge] * (1 - x_bar[edge]) * num_random_sols
        edge_correlations[edge] = sigma_c / np.sqrt(sigma_x * sol_val_var)

    # Convert into numpy matrix
    edge_ranks = sol_to_matrix(avg_edge_ranks, shape=var_costs.shape)
    edge_correlations = sol_to_matrix(edge_correlations, shape=var_costs.shape)

    # Normalize by maximum rank
    edge_ranks = edge_ranks / edge_ranks.max()
    edge_correlations = edge_correlations / edge_correlations.min()

    return edge_ranks, edge_correlations


def get_advanced_edge_features_for_instance(
    instance,
    include_statistical_features=False,
    include_node_features=True,
    batch_dim=True,
):
    """Get advanced edge features.

    Features:
    - supply_demand_ratio
    - relative capacity
    - variable costs
    - fixed costs
    - variable costs (relative)
    - fixed costs (relative)
    - variable-fixed-cost ratio
    - supply source node
    - demand target node
    - rank-based score
    - correlation-based score

    Parameters
    ----------
    instance: tuple
        Instance tuple.
    include_statistical_features: bool, optional
        Indicate whether statistical features should be included. Default is
        False.
    include_node_features: bool, optional
        Indicate whether features of source and target should be included.
        Default is True.
    batch_dim: bool, optional
        Indicate whether batch dimension should be added.

    Returns
    -------
    edge_features: np.array
        Edge features.

    """
    supply, demand, var_costs, fix_costs = (
        instance.supply,
        instance.demand,
        instance.var_costs,
        instance.fix_costs,
    )
    supply_exp = np.repeat(np.expand_dims(supply, axis=1), repeats=len(demand), axis=1)
    demand_exp = np.repeat(np.expand_dims(demand, axis=0), repeats=len(supply), axis=0)

    capacity = np.minimum(supply_exp, demand_exp)
    rel_capacity = capacity / capacity.mean()
    supply_demand_ratio = supply_exp / demand_exp
    rel_var_costs = var_costs / np.mean(var_costs)
    rel_fix_costs = fix_costs / np.mean(fix_costs)
    c_f_ratio = (var_costs * capacity) / fix_costs
    edge_features = [
        supply_demand_ratio,
        rel_capacity,
        var_costs,
        fix_costs,
        rel_var_costs,
        rel_fix_costs,
        c_f_ratio,
    ]
    if include_node_features:
        edge_features = [supply_exp, demand_exp] + edge_features

    if include_statistical_features:
        edge_ranks, edge_correlations = get_statistical_edge_features(instance)
        edge_features.append(edge_ranks)
        edge_features.append(edge_correlations)

    edge_features = np.dstack(edge_features)

    if batch_dim:
        return edge_features[None, :, :, :]
    else:
        return edge_features


def get_edge_features(sample_paths, features="combined_raw_edge_features"):
    """Translate a set of N sample instances into an edge feature array.

    Parameters
    ----------
    sample_paths: list
        List of sample paths.
    features: str, optional
        Type of features. Can be 'combined_raw_edge_features', 'bipartite_advanced_edge_features' or
        'bipartite_advanced_edge_features_plus_stat_features'.

    Returns
    -------
    np.array
        Edge features.

    """

    if features == "combined_raw_edge_features":
        feature_fun = get_combined_raw_edge_features_for_instance
    elif features == "advanced_edge_features":
        feature_fun = get_advanced_edge_features_for_instance
    elif features == "advanced_edge_features_plus_stat_features":
        feature_fun = partial(
            get_advanced_edge_features_for_instance, include_statistical_features=True
        )

    x_connection = []
    # Read samples from pickle files
    for sample_path in sample_paths:
        sample = load_sample(sample_path)
        # Prepare features
        x_connection.append(feature_fun(sample["instance"]))

    x_connection = np.concatenate(x_connection)

    return x_connection


###################################
# Label generation
###################################


def get_raw_target_for_instance(
    sol, instance=None, binary_target=True, output_dim=True, batch_dim=True
):
    """Translate solution into target for learning.

    Parameters
    ----------
    sol: np.array
        Solution matrix containing flows.
    instance: tuple
        Instance.
    binary_target: bool, optional
        Indicate whether target should be binary (1 if flow >0, 0 otherwise).
    output_dim: bool, optional
        Indicate whether output dimension should be added.
    batch_dim: bool, optional
        Indicate whether batch dimension should be added.

    Returns
    -------
    np.array
        Target array.

    """
    if output_dim:
        sol = sol[:, :, None]
    if binary_target:
        sol = np.where(sol > 0, 1, 0)
    if batch_dim:
        sol = np.expand_dims(sol, axis=0)
    return sol


def get_raw_targets(
    sample_paths,
    binary_target=True,
    output_dim=True,
):
    """Translate a set of N sample solutions into labels for learning.

    Parameters
    ----------
    sample_paths: list
        List of sample paths.
    binary_target: bool, optional
        Indicate whether target should be binary (1 if flow >0, 0 otherwise).
    batch_dim: bool, optional
        Indicate whether batch dimension should be added.

    Returns
    -------
    np.array
        Target array.

    """
    y = []

    # Read samples from pickle files
    for sample_path in sample_paths:
        sample = load_sample(sample_path)
        sol = sol_to_matrix(sample["solution"])
        # Prepare target
        if output_dim:
            sol = sol[:, :, None]
        if binary_target:
            sol = np.where(sol > 0, 1, 0)
        y.append(sol)

    y = np.array(y)

    return y


def get_class_weights(y, binary=False):
    """Calculate class weights for unbalanced data set.

    Parameters
    ----------
    y: np.array
        Labels.
    binary: bool, optional
        Indicate whether there are two or more classes.

    Returns
    -------
    float or np.array
        Class weight for positive class (binary classification) or class weights
        for all classes.

    """
    if isinstance(y, list):
        y = np.concatenate([i.flatten() for i in y])
    y = y.flatten()
    cw = compute_class_weight("balanced", classes=np.unique(y), y=y)
    if binary:
        return cw[1] / cw[0]
    else:
        return cw


###################################
# Data set classes
###################################


class FCTPData(Dataset):
    """Dataset wrapper for FCTP training data.

    Parameters
    ----------
    x: list of np.array
        List of feature arrays.
    y: np.array
        Labels.

    """

    def __init__(self, x, y):
        super(FCTPData, self).__init__()

        self.x = x
        self.y = y

        self.data_shapes = tuple([x_i[0].shape[-1] for x_i in self.x])

    def __getitem__(self, index):
        return (
            tuple([x_i[index] for x_i in self.x]),
            self.y[index],
        )

    def __len__(self):
        return len(self.y)


###################################
# Normalization and transformation functions
###################################


class StandardNormalizer:
    """Normalizer to normalize to zero mean and unit variance.

    Parameters
    ----------
    mean: np.array, optional
        Array of feature means. If not provided, it will be calculated from data in the fit
        function.
    std: np.array, optional
        Array of feature standard deviations. If not provided, it will be calculated from data in
        the fit function.

    """

    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def fit(self, data):
        """Fit normalizer.

        Parameters
        ----------
        data: np.array
            Data (num_samples x num_features).

        Returns
        -------
        None

        """
        if isinstance(data, list):
            raise NotImplementedError

        # compute mean and std for each feature (i.e., exclude feature dimension)
        dims = tuple(np.arange(len(data.shape) - 1))
        self.mean = np.mean(data, axis=dims)
        self.std = np.std(data, axis=dims)
        self.std = np.where(self.std < 1e-5, 1, self.std)  # prevent divison by zero

    def transform(self, data):
        """Apply normalization.

        Parameters
        ----------
        data: np.array
            Data (num_samples x num_features) to be normalized.

        Returns
        -------
        np.array
            Normalized data.

        """
        return (data - self.mean) / self.std

    def get_state(self):
        """Retrieve normalizer attributes.

        Returns
        -------
        tuple
            2-element tuple, containing mean and standard deviation arrays.

        """
        return (self.mean, self.std)


class MultiInputNormalizer:
    """Wrapper around multiple normalizers.

    Parameters
    ----------
    normalizers: list of StandardNormalizer
        Normalizers.

    """

    def __init__(self, normalizers):
        self._normalizers = normalizers

    def fit(self, data):
        """Fit all normalizers.

        Parameters
        ----------
        data: list of np.array
            List of data arrays (num_samples x num_features). Needs to have the same length as
            self._normalizers.

        Returns
        -------
        None

        """
        assert len(data) == len(
            self._normalizers
        ), "Number of inputs and normalizers does not match"

        for i, normalizer in enumerate(self._normalizers):
            if normalizer is not None:
                normalizer.fit(data[i])

    def transform(self, data_orig):
        """Apply normalizers.

        Parameters
        ----------
        data_orig: list of np.array
            List of data arrays (num_samples x num_features) to be normalized.
            Needs to have the same length as self._normalizers.

        Returns
        -------
        data: list of np.array
            Normalized data.

        """
        assert len(data_orig) == len(
            self._normalizers
        ), "Number of inputs and normalizers does not match"

        data = []
        for i, normalizer in enumerate(self._normalizers):
            if normalizer is not None:
                data.append(normalizer.transform(data_orig[i]))
            else:
                data.append(data_orig[i])

        return tuple(data)

    def fit_transform(self, data):
        """Fit and transform.

        Parameters
        ----------
        data: list of np.array
            List of data arrays (num_samples x num_features). Needs to have the same length as
            self._normalizers.

        Returns
        -------
        list of np.array
            Normalized data.

        """
        self.fit(data)
        return self.transform(data)

    def get_state(self):
        """Retrieve normalizer attributes.

        Returns
        -------
        tuple
            Mean and standard deviation arrays for all normalizers.

        """
        state = []
        for normalizer in self._normalizers:
            if normalizer is not None:
                state.append(normalizer.get_state())
        return tuple(state)


############################
# Training wrapper and helper functions
############################


def binary_classification_eval_display_func(p, include_train=True):
    """Print model performance of binary classification model in pretty format.

    Parameters
    ----------
    p: dict
        Model performance dictionary.
    include_train: bool, optional
        Indicate whether train data performance should be printed in addition to
        validation data performance. Default is True.

    Returns
    -------
    str
        Performance summary as string.

    """
    if include_train:
        return (
            f"Train loss: {np.round(p['train_loss'], 4)} "
            f"Train acc: {np.round(100 * p['train_accuracy'],2)}%. "
            f"Train rec: {np.round(100 * p['train_recall'],2)}%. "
            f"Validation loss: {np.round(p['validation_loss'], 4)} "
            f"Validation acc: {np.round(100 * p['validation_accuracy'],2)}%. "
            f"Validation rec: {np.round(100 * p['validation_recall'],2)}%. "
        )

    else:
        return (
            f"Validation loss: {np.round(p['validation_loss'], 4)} "
            f"Validation acc: {np.round(100 * p['validation_accuracy'],2)}%. "
            f"Validation rec: {np.round(100 * p['validation_recall'],2)}%. "
        )


def multiclass_classification_eval_display_func(p, include_train=True):
    """Print model performance of multiclass classification model in pretty format.

    Parameters
    ----------
    p: dict
        Model performance dictionary.
    include_train: bool, optional
        Indicate whether train data performance should be printed in addition to
        validation data performance. Default is True.

    Returns
    -------
    str
        Performance summary as string.

    """
    if include_train:
        return (
            f"Train loss: {np.round(p['train_loss'], 4)} "
            f"Train acc: {np.round(100 * p['train_accuracy'],2)}%. "
            f"Validation loss: {np.round(p['validation_loss'], 4)} "
            f"Validation acc: {np.round(100 * p['validation_accuracy'],2)}%. "
        )

    else:
        return (
            f"Validation loss: {np.round(p['validation_loss'], 4)} "
            f"Validation acc: {np.round(100 * p['validation_accuracy'],2)}%. "
        )


def save_checkpoint(
    policy, exp_dict, epoch, model_config, training_config, chkpnt_path
):
    """Save pytorch model parameters, experiment dict and configs.

    Parameters
    ----------
    policy: BaseSolEdgePredictor
        FCTP solution edge predictor.
    exp_dict: dict or defaultdict
        Experiment dictionary.
    epoch: int
        Training epoch.
    model_config: dict
        Model config dict.
    training_config: dict
        Training config dict.
    chkpnt_path: str
        Path to save checkpoint.

    Returns
    -------
    None

    """
    if policy.input_transformer is not None:
        normalizer_state = policy.input_transformer.get_state()
    else:
        normalizer_state = None
    torch.save(
        {
            "epoch": epoch,
            "model_config": model_config,
            "state_dict": policy.model.state_dict(),
            "normalizers": normalizer_state,
            "training_config": training_config,
            "exp_dict": default_to_regular_dict(exp_dict),
        },
        chkpnt_path,
    )


def training_wrapper(
    training_config,
    model_config,
    train_dataset,
    val_dataset,
    policy,
    out_dir,
    logger,
    eval_train_data=False,
    initial_eval_display_func=None,
    running_eval_display_func=None,
):
    """Model training wrapper function.

    Parameters
    ----------
    training_config: dict
        Training config dict.
    model_config: dict
        Model config dict.
    train_dataset: FCTPData
        Training data.
    val_dataset: FCTPData
        Validation data.
    policy: BaseSolEdgePredictor
        FCTP solution edge predictor to be trained.
    out_dir: str
        Path to output directory to save checkpoints and logging file.
    logger: logging.Logger
        Logger object.
    eval_train_data: bool, optional
        Indicate whether training data performance should be evaluated after every epoch. Default
        is False.
    initial_eval_display_func: Function
        Function to display initial model performance as pretty string.
    running_eval_display_func: Function
        Function to display model performance after every epoch as pretty string.

    Returns
    -------
    exp_dict: dict or defaultdict
        Experiment dictionary.
    best_period: int
        Epoch of best model performance.
    best_kpi_val: float
        Value of monitoring KPI (loss, accuracy, etc.) in best epoch.

    """
    if initial_eval_display_func is None:
        initial_eval_display_func = lambda x: x

    if running_eval_display_func is None:
        running_eval_display_func = lambda x: x

    # Training
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=training_config.train_batch_size,
        shuffle=True,
    )

    # Validation
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=training_config.val_batch_size,
        shuffle=False,
    )

    logger.info(f"Number of training samples: {len(train_dataset)}")
    logger.info(f"Number of validation samples: {len(val_dataset)}")

    #######################################################
    # Training Loop
    #######################################################

    # Pre-configure saving function
    save_chkpnt = partial(
        save_checkpoint,
        model_config=model_config,
        training_config=dict(training_config),
    )

    exp_dict = defaultdict(lambda: [])

    exp_dict["device"] = "gpu" if training_config.use_gpu else "cpu"
    exp_dict["num_train_samples"] = len(train_dataset)
    exp_dict["max_num_epochs"] = training_config.max_num_epochs
    exp_dict["initial_performance"] = defaultdict(lambda: [])
    exp_dict["performance"] = defaultdict(lambda: [])

    # Evaluate policy before first update
    p = policy.evaluate({"train": train_loader, "validation": val_loader})
    for key, value in p.items():
        exp_dict["initial_performance"][key].append(value)
    logger.info(("[Initial performance] " f"{initial_eval_display_func(p)}"))

    # Training loop
    best_period = 0
    best_kpi_val = np.inf
    total_start_time = time.time()
    logger.info("Starting training loop...")
    for i in range(training_config.max_num_epochs):

        policy.model.train()

        start_time = time.time()
        start_time_process = time.process_time()

        running_loss = 0
        for batch in train_loader:
            # Update model parameters
            loss, _ = policy.train_step(batch)
            running_loss += loss.cpu().item()
        running_loss = running_loss / len(train_loader)

        train_time = time.time() - start_time
        train_time_process = time.process_time() - start_time_process
        exp_dict["train_time"].append(train_time)
        exp_dict["train_time_process"].append(train_time_process)
        exp_dict["performance"]["train_loss"].append(running_loss)

        # Evaluate model performance
        if eval_train_data:
            p = policy.evaluate({"train": train_loader, "validation": val_loader})
        else:
            p = policy.evaluate({"validation": val_loader})
        for key, value in p.items():
            exp_dict["performance"][key].append(value)

        # Adapt learning rate based on performance stagnation
        policy.lr_scheduler_step(p[training_config.monitoring_kpi])

        # Log current epoch performance and save checkpoint
        logger.info(
            (
                f"[Epoch {i+1}/{training_config.max_num_epochs}] "
                f"Running train loss: {np.round(running_loss, 4)} "
                f"{running_eval_display_func(p)} "
                f"({datetime.timedelta(seconds=train_time)})"
            )
        )
        save_chkpnt(
            policy,
            exp_dict,
            i + 1,
            chkpnt_path=os.path.join(out_dir, "checkpoint.pth.tar"),
        )

        # Check if model performance has improved
        if p[training_config.monitoring_kpi] < best_kpi_val * (
            1 - training_config.opt_threshold
        ):
            logger.info("Better model found!")
            best_kpi_val = p[training_config.monitoring_kpi]
            best_period = i
            save_chkpnt(
                policy,
                exp_dict,
                i + 1,
                chkpnt_path=os.path.join(out_dir, "best_checkpoint.pth.tar"),
            )

        # Early stopping if model performance has not improved over several epochs
        if i - best_period >= training_config.early_stopping:
            logger.info(f"No improvement since period {best_period + 1} -> Stopping")
            break

    total_train_time = time.time() - total_start_time
    logger.info(
        f"Training finished after {datetime.timedelta(seconds=total_train_time)}."
    )
    return exp_dict, best_period, best_kpi_val


############################
# Model pre- and postprocessing
############################


def load_edge_predictor_model(model_path, get_feature_fun=True, get_label_fun=False):
    """Helper function to load edge predictor model.

    Parameters
    ----------
    model_path: str
        Path to model checkpoint, containing model weights, normalizer states, and configs.
    get_feature_fun: bool, optional
        Indicate whether feature extraction function should be returned in addition to edge
        predictor model. Default is True.
    get_label_fun: bool, optional
        Indicate whether label extraction function should be returned in addition to edge predictor
        model. Default is True.

    Returns
    -------
    tuple
        1-element, 2-element, or 3-element tuple, comprising edge predictor model and potentially
        feature extraction function and label extraction function.

    """
    return_vals = []

    chkpnt = torch.load(model_path, map_location="cpu")
    model_config = chkpnt["model_config"]
    model_state = chkpnt["state_dict"]
    normalizer_state = chkpnt["normalizers"]

    # Configure policy
    if model_config.model == "gcnn":
        policy_fun = GCNNSolEdgePredictor
    elif model_config.model == "linear_logreg":
        policy_fun = EdgeLogRegSolEdgePredictor
    elif model_config.model == "edge_mlp":
        policy_fun = EdgeMLPSolEdgePredictor
    else:
        raise ValueError

    # Configure normalizer
    input_transformer = None
    if model_config.normalization == "standard":
        assert (
            normalizer_state is not None
        ), "Normalization parameters expected but are None"
        normalizers = tuple([StandardNormalizer(*vals) for vals in normalizer_state])
        input_transformer = MultiInputNormalizer(normalizers)

    # Load model and parameters
    policy = policy_fun(
        model_config=model_config,
        input_transformer=input_transformer,
    )
    policy.model.load_state_dict(model_state)
    policy.model.eval()

    if not get_feature_fun and not get_label_fun:
        return policy

    return_vals = [policy]

    if get_feature_fun:
        # Configure feature extraction function
        if model_config.features == "bipartite_raw":
            feature_fun = get_bipartite_raw_features_for_instance
        elif model_config.features == "combined_raw_edge_features":
            feature_fun = get_combined_raw_edge_features_for_instance
        elif model_config.features == "advanced_edge_features":
            feature_fun = get_advanced_edge_features_for_instance
        elif model_config.features == "advanced_edge_features_plus_stat_features":
            feature_fun = partial(
                get_advanced_edge_features_for_instance,
                include_statistical_features=True,
            )
        elif model_config.features == "bipartite_advanced_edge_features":
            feature_fun = get_bipartite_advanced_edge_features_for_instance
        elif (
            model_config.features
            == "bipartite_advanced_edge_features_plus_stat_features"
        ):
            feature_fun = partial(
                get_bipartite_advanced_edge_features_for_instance,
                include_statistical_features=True,
            )
        else:
            raise ValueError
        return_vals.append(feature_fun)

    if get_label_fun:
        # Configure label extraction function
        if model_config.prediction_task == "binary_classification":
            label_fun = partial(get_raw_target_for_instance, binary_target=True)
        else:
            raise ValueError
        return_vals.append(label_fun)

    return tuple(return_vals)
