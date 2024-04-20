""" TP and FCTP heuristics. """

from copy import deepcopy
import numpy as np

from core.utils.preprocessing import get_weight_matrix
from core.utils.preprocessing import balance_tp


def north_west_corner(supplies, demands):
    """North-West-Corner Method for (FC)TP.

    Parameters
    ----------
    supplies: 1D np.array or list
        A list of supplier capacities/supplies.
    demands: 1D np.array or list
        A list of customer demands.

    Returns
    -------
    sol: dict
        Flow from supplier i to customer j.

    """

    assert supplies.sum() == demands.sum(), "Total supply does not equal total demand"
    supplies_copy = supplies.copy()
    demands_copy = demands.copy()
    i = 0
    j = 0
    bfs = {}
    while len(bfs) < len(supplies) + len(demands) - 1:
        s = supplies_copy[i]
        d = demands_copy[j]
        v = min(s, d)
        supplies_copy[i] -= v
        demands_copy[j] -= v
        bfs[(i, j)] = v
        if supplies_copy[i] == 0 and i < len(supplies) - 1:
            i += 1
        elif demands_copy[j] == 0 and j < len(demands) - 1:
            j += 1
    return bfs


def least_cost_method(supplies, demands, costs):
    """Least-Cost-Method for TP.

    Parameters
    ----------
    supplies: 1D np.array or list
        A list of supplier capacities/supplies.
    demands: 1D np.array or list
        A list of customer demands.
    costs: 2D np.array
        Costs for sending one unit of flow from supplier i to customer j.

    Returns
    -------
    sol: dict
        Flow from supplier i to customer j.

    """

    supplies = deepcopy(supplies)
    demands = deepcopy(demands)
    costs = deepcopy(costs)

    # original indices of suppliers and customers
    X = np.arange(len(supplies))
    Y = np.arange(len(demands))

    sol = dict()
    while len(demands) > 0:
        # find smallest entry and cost matrix to determine supplier and customer
        i, j = np.unravel_index(costs.argmin(), costs.shape)
        # assign flow, update solution, supply, demand and cost matrix
        # case 1: supply is smaller than demand -> assign entire supply, remove supplier and update
        # demand
        if supplies[i] < demands[j]:
            demands[j] -= supplies[i]
            sol[(X[i], Y[j])] = supplies[i]
            X = np.delete(X, i)
            supplies = np.delete(supplies, i)
            costs = np.delete(costs, i, axis=0)
        # case 2: supply is larger than demand -> fulfill entire demand, remove customer and update
        # supply
        elif supplies[i] > demands[j]:
            supplies[i] -= demands[j]
            sol[(X[i], Y[j])] = demands[j]
            Y = np.delete(Y, j)
            demands = np.delete(demands, j)
            costs = np.delete(costs, j, axis=1)
        # case 3: supply = demand -> assign entire supply, remove supplier and customer
        else:
            sol[(X[i], Y[j])] = supplies[i]
            X = np.delete(X, i)
            Y = np.delete(Y, j)
            supplies = np.delete(supplies, i)
            demands = np.delete(demands, j)
            costs = np.delete(costs, i, axis=0)
            costs = np.delete(costs, j, axis=1)
    return sol


def lcm_fctp(instance, costs=None, only_var_costs=False):
    """Convert FCTP into TP (relaxation) and solve via LCM.

    Parameters
    ----------
    instance: FCTP
        Instance to solve.
    costs: np.array, optional
        Costs for sending flow from supplier i to customer j. If not provided, it will be calculated
        from variable and fixed costs.
    only_var_costs: bool, optional
        Indicate whether only variable costs should be considered in the relaxed TP. Default is
        False.

    Returns
    -------
    sol: dict
        Flow from supplier i to customer j.

    """
    if costs is None:
        # combine variable and fixed costs into a single cost value
        costs = get_weight_matrix(
            instance.var_costs,
            instance.fix_costs,
            instance.supply,
            instance.demand,
            only_var_costs=only_var_costs,
            inverse=False,
        )

    # add dummy demand if FCTP is unbalanced (excess supply)
    supplies = instance.supply
    demands = instance.demand
    dummy_demand_added, demands, costs = balance_tp(
        supplies, demands, costs, minimization=True
    )

    # solve TP vias LCM
    sol = least_cost_method(supplies, demands, costs)

    # remove dummy-flows from solution (i.e., flows to last customer)
    if dummy_demand_added:
        sol = {(i, j): v for (i, j), v in sol.items() if j < len(demands) - 1}

    return sol


def k_nearest_supplier_predictor(instance, k=5):
    """Binary adjacency matrix indicating whether a supplier is a k-nearest supplier.

    Parameters
    ----------
    instance: FCTP
        Instance to solve.
    k: int, optional
        The number of closest suppliers that should be included for each customer.

    Returns
    -------
    relevant_connections: 2D np.array
        Adjacency matrix where a 1 indicates that the supplier is a k-nearest supplier for the
        respective customer.

    """
    W = get_weight_matrix(
        instance.var_costs,
        instance.fix_costs,
        instance.supply,
        instance.demand,
        inverse=False,
    )
    cheapest_supply_ind = np.argsort(W, axis=0)
    k_nearest_supplier = cheapest_supply_ind[:k, :]
    relevant_connections = np.full(W.shape, False, dtype=bool)
    # np.take_along_axis(relevant_connections, k_nearest_supplier, axis=0)
    for j in range(instance.n):
        for i in k_nearest_supplier[:, j]:
            relevant_connections[i, j] = True
    return relevant_connections


def k_shortest_edges_predictor(instance, k=50):
    """Binary adjacency matrix indicating whether an edge is among the k-cheapest edges.

    Parameters
    ----------
    instance: FCTP
        Instance to solve.
    k: int, optional
        The number of closest suppliers that should be included for each customer.

    Returns
    -------
    relevant_connections: 2D np.array
        Adjacency matrix where a 1 indicates that the edge is among the k-cheapest edges.

    """
    W = get_weight_matrix(
        instance.var_costs,
        instance.fix_costs,
        instance.supply,
        instance.demand,
        inverse=False,
    )
    cheapest_ind = np.unravel_index(np.argsort(W, axis=None), W.shape)
    relevant_connections = np.full(W.shape, False, dtype=bool)
    # np.take_along_axis(relevant_connections, k_nearest_supplier, axis=0)
    for r in range(k):
        i = cheapest_ind[0][r]
        j = cheapest_ind[1][r]
        relevant_connections[i, j] = True
    return relevant_connections


def random_edges_predictor(shape, k=50):
    """Binary adjacency matrix indicating whether an edge is randomly selected into the edge set.

    Parameters
    ----------
    shape: tuple
        Shape of the adjacency matrix.
    k: int, optional
        The number of edges that should be selected.

    Returns
    -------
    relevant_connections: 2D np.array
        Adjacency matrix where a 1 indicates that the edge is among the k selected edges.

    """
    n = np.prod(shape)
    relevant_connections = np.full(n, False, dtype=bool)
    indices = np.random.choice(np.arange(n), size=k, replace=False)
    relevant_connections[indices] = True
    relevant_connections = relevant_connections.reshape(shape)
    return relevant_connections


def add_feasible_sol_connections(connections, instance, method="lcm"):
    """Solve FCTP via LCM and add solution edges to a given set of connections.

    Parameters
    ----------
    connections: 2D np.array or dict
        Binary matrix or dictionary indicating whether a certain edge is already included.
    instance: FCTP
        Instance to be solved.
    method: str, optional
        Method to generate feasible solution ("lcm", "nwc"). Default is "lcm".

    Returns
    -------
    connections: 2D np.array or dict
        Binary matrix or dictionary indicating whether a certain edge is included in the set of
        connections.

    """
    connections = deepcopy(connections)
    if method == "lcm":
        sol = lcm_fctp(instance)
    elif method == "nwc":
        sol = north_west_corner(instance.supply, instance.demand)
    else:
        raise ValueError
    for (i, j), val in sol.items():
        if abs(val) > 1e-5:
            connections[i, j] = True
    return connections
