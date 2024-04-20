""" Utility functions for pre-processing FCTP data. """

from copy import deepcopy

import networkx as nx
import numpy as np

from core.utils.postprocessing import matrix_to_dict
from core.utils.fctp import FCTP


def min_of_supply_demand(supplies, demands):
    """Calculate for each edge (i,j) the minimum of s_i and d_j.

    Parameters
    ----------
    supplies: 1D np.array or list
        A list of supplier capacities/supplies.
    demands: 1D np.array or list
        A list of customer demands.

    Returns
    -------
    2D np.array
        Edge matrix containing the minimum of s_i and d_j for each edge (i,j).

    """
    supply_exp = np.repeat(
        np.expand_dims(supplies, axis=1), repeats=len(demands), axis=1
    )
    demand_exp = np.repeat(
        np.expand_dims(demands, axis=0), repeats=len(supplies), axis=0
    )
    return np.minimum(supply_exp, demand_exp)


def get_weight_matrix(
    var_costs, fix_costs, supplies, demands, only_var_costs=False, inverse=True
):
    """Linearize FCTP costs.

    w_ij = c_ij + f_ij / min{s_i, d_j}

    Parameters
    ----------
    var_costs: 2D np.array
        Variables costs for sending one unit of flow form supplier i to customer j.
    fix_costs: 2D np.array
        Fixed costs for sending flow form supplier i to customer j.
    supplies: 1D np.array or list
        A list of supplier capacities/supplies.
    demands: 1D np.array or list
        A list of customer demands.
    only_var_costs: bool, optional
        Indicate whether only variable costs should be included. Default is False.
    inverse: bool, optional
        Indicate whether the inverse should be returned (maximization). Default is True.

    Returns
    -------
    W: 2D np.array
        Edge weight matrix.

    """
    min_s_d = min_of_supply_demand(supplies, demands)

    if only_var_costs:
        W = var_costs
    else:
        W = var_costs + fix_costs / min_s_d

    if inverse:
        W = 1 / W

    return W


def balance_tp(supplies, demands, costs, minimization=True):
    """Helper function to balance supply and demand in case of excess supply.

    Parameters
    ----------
    supplies: 1D np.array or list
        A list of supplier capacities/supplies.
    demands: 1D np.array or list
        A list of customer demands.
    costs: 2D np.array
        Costs for sending flow from supplier i to customer j.
    minimization: bool, optional
        Indicate if it is a minimization or maximization problem. Default is
        minimization (True).

    Returns
    -------
    dummy_demand_added: bool
        True if dummy demand has been added, false otherwise.
    demands: 1D np.array or list
        Updated demand vector containing dummy demand.
    costs: 2D np.array
        Updated cost matrix containing costs to dummy demand node.

    """
    dummy_demand_added = False
    delta = np.sum(supplies) - np.sum(demands)
    assert delta >= 0, "Total demand must not be larger than total supply"
    if delta > 0:
        dummy_demand_added = True
        dummy_cost = 1e6 if minimization else -1e6
        # add dummy demand node that covers the delta
        demands = np.append(demands, delta)
        costs = np.append(costs, np.full((costs.shape[0], 1), dummy_cost), axis=1)
    assert np.sum(supplies) == np.sum(demands)
    return dummy_demand_added, demands, costs


def get_tp_graph(relevant_connections):
    """Translate set of TP connections into networkx graph.

    Parameters
    ----------
    relevant_connections: 2D np.array
        Binary edge matrix indicating relevant connections.

    Returns
    -------
    graph: nx.Graph
        TP graph including specified connections.

    """
    edges = list(matrix_to_dict(relevant_connections, include_zeros=False).keys())
    edges = [(f"S{i}", f"D{j}") for i, j in edges]
    graph = nx.Graph()
    graph.add_edges_from(edges)
    return graph


def get_subproblems(instance, relevant_connections):
    """Extract disjoint subproblems based on TP connections.

    Parameters
    ----------
    instance: FCTP
        Original FCTP problem.
    relevant_connections: 2D np.array
        Binary edge matrix indicating relevant connections.

    Returns
    -------
    subproblems: list
        List of disjoint subproblems.

    """
    tp_graph = get_tp_graph(relevant_connections)
    # Find connected components
    ccs = list(nx.connected_components(tp_graph))
    if len(ccs) == 1:
        return [
            (
                np.arange(instance.m),
                np.arange(instance.n),
                deepcopy(instance),
                relevant_connections,
            )
        ]
    # Treat each connected component as a disjoint subproblem
    subproblems = []
    for cc in ccs:
        cc = list(cc)
        s_nodes = []
        d_nodes = []
        for node in cc:
            node_type = node[0]
            node_id = int(node[1:])
            if node_type == "S":
                s_nodes.append(node_id)
            else:
                d_nodes.append(node_id)
        s_nodes.sort()
        d_nodes.sort()
        subproblems.append(
            (
                s_nodes,
                d_nodes,
                FCTP(
                    supply=instance.supply[s_nodes],
                    demand=instance.demand[d_nodes],
                    var_costs=instance.var_costs[s_nodes, :][:, d_nodes],
                    fix_costs=instance.fix_costs[s_nodes, :][:, d_nodes],
                ),
                relevant_connections[s_nodes, :][:, d_nodes],
            )
        )
    return subproblems
