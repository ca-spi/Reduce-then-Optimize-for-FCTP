""" Utility functions for post-processing solutions. """

from collections import defaultdict

import networkx as nx
import numpy as np


def merge_solutions(sols, keys=None):
    """Combine multiple solutions by summing up flows on same connections.

    Parameters
    ----------
    sols: list
        A list of solution dictionaries.

    Keys: list, optional
        A list of keys that should be contained in the new solution dict. Otherwise, only the keys
        from the individual solution dictionaries are considered.

    Returns
    -------
    merged_sol: dict
        Total flow from supplier i to customer j.

    """
    if keys:
        merged_sol = {k: 0 for k in keys}
    else:
        merged_sol = defaultdict(0)
    for sol in sols:
        for k, v in sol.items():
            # sum up flows on same connection
            merged_sol[k] += v
    return merged_sol


def sol_to_matrix(solution, shape=None):
    """Convert a solution-dict into a solution-matrix.

    Parameters
    ----------
    solution: dict
        Solution dictionary containing flows from supplier i to customer j.
    shape: tuple, optional
        Target shape. If it is not provided, it is derived from the dictionary keys.

    Returns
    -------
    matrix: 2D np.array
        Solution matrix containing flows from supplier i to customer j.

    """
    if shape is None:
        # derive shape from largest supplier and customer key
        keys = list(solution.keys())
        n_suppliers = max([k[0] for k in keys]) + 1
        n_customers = max([k[1] for k in keys]) + 1
        shape = (n_suppliers, n_customers)
    matrix = np.zeros(shape)
    for (i, j), v in solution.items():
        matrix[i, j] = v
    return matrix


def matrix_to_dict(matrix, include_zeros=True):
    """Convert a solution-matrix into a solution-dict.

    Parameters
    ----------
    matrix: 2D np.array
        Solution matrix containing flows from supplier i to customer j.
    include_zeros: bool, optional
        Indicate whether zero-flows should be included in solution dictionary. Default is True.

    Returns
    -------
    dict
        Solution dictionary containing flows from supplier i to customer j.

    """
    n_suppliers, n_customers = matrix.shape
    if include_zeros:
        return {
            (i, j): matrix[i, j] for i in range(n_suppliers) for j in range(n_customers)
        }
    else:
        return {
            (i, j): matrix[i, j]
            for i in range(n_suppliers)
            for j in range(n_customers)
            if matrix[i, j] != 0
        }


def sol_to_graph(solution):
    """Convert a solution-dict into a networkx-graph.

    Parameters
    ----------
    solution: dict
        Solution dictionary containing flows from supplier i to customer j.

    Returns
    -------
    graph: nx.Graph
        Undirected graph containing flows from suppliers to customers.

    """
    keys = list(solution.keys())
    suppliers = {k[0] for k in keys}
    customers = {k[1] for k in keys}

    graph = nx.Graph()
    graph.add_nodes_from(["supplier_{}".format(i) for i in suppliers])
    graph.add_nodes_from(["customer_{}".format(i) for i in customers])
    graph.add_edges_from(
        [
            (
                "supplier_{}".format(i),
                "customer_{}".format(j),
            )
            for (i, j), v in solution.items()
            if v > 0
        ]
    )

    return graph


def reindex_sol(sol, orig_ids_supply, orig_ids_demand):
    """Reindex solution.

    Parameters
    ----------
    sol: dict
        Solution dictionary containing flows from i to j.
    orig_ids_supply: list
        Mapping of solution supplier IDs to original IDs.
    orig_ids_demand: list
        Mapping of solution customer IDs to original IDs.

    Returns
    -------
    dict
        Solution dictionary with re-mapped keys.

    """
    return {(orig_ids_supply[i], orig_ids_demand[j]): k for (i, j), k in sol.items()}
