""" LP/MIP models and helper functions to solve TP and FCTP instances. """

import random

import gurobipy as gp
import networkx as nx
import numpy as np

from core.utils.preprocessing import get_weight_matrix

##################################################
# LP and MIP Models
##################################################


def tp(supplies, demands, costs):
    """Generate Gurobi model for the TP.

    Parameters
    ----------
    supplies: 1D np.array or list
        A list of supplier capacities/supplies.
    demands: 1D np.array or list
        A list of customer demands.
    costs: 2D np.array
        Costs for sending one unit of flow form supplier i to customer j.

    Returns
    -------
    m: gp.Model
        Gurobi model of TP instance.
    x: dict
        Dictionary of flow variables, keyed by (supplier, customer).

    """

    m = gp.Model()

    # The flow variables represent the absolute flow from a certain supplier to a certain customer
    x = m.addVars(*costs.shape, obj=costs, vtype=gp.GRB.CONTINUOUS, name="x")

    # For each customer, ensure that the total demand is fulfilled
    for j, demand in enumerate(demands):
        m.addConstr(
            gp.quicksum(x[i, j] for i in range(len(supplies)) if (i, j) in x) == demand,
            name=f"B{j}",
        )

    # For each supplier, ensure that the total capacity is not exceeded (unbalanced) or
    # fully distributed (balanced)
    balanced = sum(supplies) == sum(demands)
    if balanced:
        for i, supply in enumerate(supplies):
            m.addConstr(
                gp.quicksum(x[i, j] for j in range(len(demands)) if (i, j) in x)
                == supply,
                name=f"A{i}",
            )
    else:
        for i, supply in enumerate(supplies):
            m.addConstr(
                gp.quicksum(x[i, j] for j in range(len(demands)) if (i, j) in x)
                <= supply,
                name=f"A{i}",
            )

    return m, x


def fctp(supplies, demands, var_costs, fix_costs, relax=False):
    """Generate Gurobi model for the FCTP.

    The FCTP extends the TP by including fixed costs on edges.

    Parameters
    ----------
    supplies: 1D np.array or list
        A list of supplier capacities/supplies.
    demands: 1D np.array or list
        A list of customer demands.
    var_costs: 2D np.array
        Variables costs for sending one unit of flow form supplier i to customer j.
    fix_costs: 2D np.array
        Fixed costs for sending flow form supplier i to customer j.
    relax: bool, optional
        Indicate whether the fixed cost variables should be linearized. Default is False.

    Returns
    -------
    m: gp.Model
        Gurobi model of FCTP instance.
    x: dict
        Dictionary of flow variables, keyed by (supplier, customer).
    y: dict
        Dictionary of activation variables, keyed by (supplier, customer).

    """

    # Get TP model
    m, x = tp(supplies, demands, var_costs)

    # Add activation variables that indicate whether a certain connection is used
    if relax:
        y = m.addVars(
            *fix_costs.shape, obj=fix_costs, vtype=gp.GRB.CONTINUOUS, name="y"
        )
    else:
        y = m.addVars(*fix_costs.shape, obj=fix_costs, vtype=gp.GRB.BINARY, name="y")

    # Set activation variable to one if the connection is used by some flow
    for i, j in x.keys():
        big_m = min(supplies[i], demands[j])
        m.addConstr(x[i, j] <= big_m * y[i, j], name=f"G[{i},{j}]")

    return m, x, y


def capacitated_fctp(
    supplies, demands, var_costs, fix_costs, edge_capacities, relax=False
):
    """Generate Gurobi model for the FCTP with edge capacities.

    Parameters
    ----------
    supplies: 1D np.array or list
        A list of supplier capacities/supplies.
    demands: 1D np.array or list
        A list of customer demands.
    var_costs: 2D np.array
        Variables costs for sending one unit of flow form supplier i to customer j.
    fix_costs: 2D np.array
        Fixed costs for sending flow form supplier i to customer j.
    edge_capacities: 2D np.array
        Maximum flow on edge.
    relax: bool, optional
        Indicate whether the fixed cost variables should be linearized. Default is False.

    Returns
    -------
    m: gp.Model
        Gurobi model of FCTP instance.
    x: dict
        Dictionary of flow variables, keyed by (supplier, customer).
    y: dict
        Dictionary of activation variables, keyed by (supplier, customer).

    """

    # Get FCTP IP
    m, x, y = fctp(supplies, demands, var_costs, fix_costs, relax)

    # Add constraint to ensure that edge capacities are not exceeded
    for i, j in x.keys():
        m.addConstr(x[i, j] <= edge_capacities[i, j], name=f"C[{i},{j}]")

    return m, x, y


def fixed_step_fctp(
    supplies, demands, var_costs, fix_costs, vehicle_capacities, relax=False
):
    """Generate Gurobi model for the FCTP with vehicle capacities and fixed-step costs.

    Parameters
    ----------
    supplies: 1D np.array or list
        A list of supplier capacities/supplies.
    demands: 1D np.array or list
        A list of customer demands.
    var_costs: 2D np.array
        Variables costs for sending one unit of flow form supplier i to customer j.
    fix_costs: 2D np.array
        Fixed costs for sending flow form supplier i to customer j.
    vehicle_capacities: 2D np.array
        Maximum quantity per vehicle.
    relax: bool, optional
        Indicate whether the fixed cost variables should be linearized. Default is False.

    Returns
    -------
    m: gp.Model
        Gurobi model of FCTP instance.
    x: dict
        Dictionary of flow variables, keyed by (supplier, customer).
    y: dict
        Dictionary of activation variables, keyed by (supplier, customer).

    """
    # Get TP model
    m, x = tp(supplies, demands, var_costs)

    # Add vehicle counter variables
    if relax:
        y = m.addVars(
            *fix_costs.shape, obj=fix_costs, vtype=gp.GRB.CONTINUOUS, name="y"
        )
    else:
        y = m.addVars(*fix_costs.shape, obj=fix_costs, vtype=gp.GRB.INTEGER, name="y")

    for i, j in x.keys():
        m.addConstr(x[i, j] <= vehicle_capacities[i, j] * y[i, j], name=f"G[{i},{j}]")

    return m, x, y


def tp_subset_connections(supplies, demands, costs, connections):
    """Generate Gurobi model for the TP with a reduced subset of connections.

    Parameters
    ----------
    supplies: 1D np.array or list
        A list of supplier capacities/supplies.
    demands: 1D np.array or list
        A list of customer demands.
    costs: 2D np.array
        Costs for sending one unit of flow form supplier i to customer j.
    connections: 2D np.array
        Binary adjacency matrix indicating whether connection should be included.

    Returns
    -------
    m: gp.Model
        Gurobi model of TP instance.
    x: dict
        Dictionary of flow variables, keyed by (supplier, customer).

    """

    m = gp.Model()

    x = dict()
    for i, _ in enumerate(supplies):
        for j, _ in enumerate(demands):
            if connections[i, j]:
                # The flow variables represent the absolute flow from a certain supplier to a certain customer
                x[i, j] = m.addVar(
                    obj=costs[i, j], vtype=gp.GRB.CONTINUOUS, name=f"x[{i},{j}]"
                )

    # For each customer, ensure that the total demand is fulfilled
    for j, demand in enumerate(demands):
        m.addConstr(
            gp.quicksum(x[i, j] for i in range(len(supplies)) if (i, j) in x) == demand,
            name=f"B{j}",
        )

    # For each supplier, ensure that the total capacity is not exceeded
    balanced = sum(supplies) == sum(demands)
    if balanced:
        for i, supply in enumerate(supplies):
            m.addConstr(
                gp.quicksum(x[i, j] for j in range(len(demands)) if (i, j) in x)
                == supply,
                name=f"A{i}",
            )
    else:
        for i, supply in enumerate(supplies):
            m.addConstr(
                gp.quicksum(x[i, j] for j in range(len(demands)) if (i, j) in x)
                <= supply,
                name=f"A{i}",
            )

    return m, x


def fctp_subset_connections(
    supplies, demands, var_costs, fix_costs, connections, relax=False
):
    """Generate Gurobi model for the FCTP with a reduced subset of connections.

    Parameters
    ----------
    supplies: 1D np.array or list
        A list of supplier capacities/supplies.
    demands: 1D np.array or list
        A list of customer demands.
    var_costs: 2D np.array
        Variables costs for sending one unit of flow form supplier i to customer j.
    fix_costs: 2D np.array
        Fixed costs for sending flow form supplier i to customer j.
    connections: 2D np.array
        Binary adjacency matrix indicating whether connection should be included.
    relax: bool, optional
        Indicate whether the fixed cost variables should be linearized. Default is False.

    Returns
    -------
    m: gp.Model
        Gurobi model of FCTP instance.
    x: dict
        Dictionary of flow variables, keyed by (supplier, customer).
    y: dict
        Dictionary of activation variables, keyed by (supplier, customer).

    """

    m, x = tp_subset_connections(supplies, demands, var_costs, connections)

    y = dict()
    for i, _ in enumerate(supplies):
        for j, _ in enumerate(demands):
            if connections[i, j]:
                # The activation variables indicate whether a certain connection is used
                if relax:
                    y[i, j] = m.addVar(
                        obj=fix_costs[i, j], vtype=gp.GRB.CONTINUOUS, name=f"y[{i},{j}]"
                    )
                else:
                    y[i, j] = m.addVar(
                        obj=fix_costs[i, j], vtype=gp.GRB.BINARY, name=f"y[{i},{j}]"
                    )

    # Set activation variable to one if the connection is used by some flow
    for i, j in x.keys():
        big_m = min(supplies[i], demands[j])
        m.addConstr(x[i, j] <= big_m * y[i, j])

    return m, x, y


def capacitated_fctp_subset_connections(
    supplies, demands, var_costs, fix_costs, edge_capacities, connections, relax=False
):
    """Generate Gurobi model for the FCTP with edge capacities.

    Parameters
    ----------
    supplies: 1D np.array or list
        A list of supplier capacities/supplies.
    demands: 1D np.array or list
        A list of customer demands.
    var_costs: 2D np.array
        Variables costs for sending one unit of flow form supplier i to customer j.
    fix_costs: 2D np.array
        Fixed costs for sending flow form supplier i to customer j.
    edge_capacities: 2D np.array
        Maximum flow on edge.
    connections: 2D np.array
        Binary adjacency matrix indicating whether connection should be included.
    relax: bool, optional
        Indicate whether the fixed cost variables should be linearized. Default is False.

    Returns
    -------
    m: gp.Model
        Gurobi model of FCTP instance.
    x: dict
        Dictionary of flow variables, keyed by (supplier, customer).
    y: dict
        Dictionary of activation variables, keyed by (supplier, customer).

    """

    # Get FCTP IP
    m, x, y = fctp_subset_connections(
        supplies, demands, var_costs, fix_costs, connections, relax
    )

    # Add constraint to ensure that edge capacities are not exceeded
    for i, j in x.keys():
        m.addConstr(x[i, j] <= edge_capacities[i, j], name=f"C[{i},{j}]")

    return m, x, y


def fixed_step_fctp_subset_connections(
    supplies,
    demands,
    var_costs,
    fix_costs,
    vehicle_capacities,
    connections,
    relax=False,
):
    """Generate Gurobi model for the FCTP with a reduced subset of connections.

    Parameters
    ----------
    supplies: 1D np.array or list
        A list of supplier capacities/supplies.
    demands: 1D np.array or list
        A list of customer demands.
    var_costs: 2D np.array
        Variables costs for sending one unit of flow form supplier i to customer j.
    fix_costs: 2D np.array
        Fixed costs for sending flow form supplier i to customer j.
    vehicle_capacities: 2D np.array
        Maximum quantity per vehicle.
    connections: 2D np.array
        Binary adjacency matrix indicating whether connection should be included.
    relax: bool, optional
        Indicate whether the fixed cost variables should be linearized. Default is False.

    Returns
    -------
    m: gp.Model
        Gurobi model of FCTP instance.
    x: dict
        Dictionary of flow variables, keyed by (supplier, customer).
    y: dict
        Dictionary of activation variables, keyed by (supplier, customer).

    """

    m, x = tp_subset_connections(supplies, demands, var_costs, connections)

    y = dict()
    for i, _ in enumerate(supplies):
        for j, _ in enumerate(demands):
            if connections[i, j]:
                # The activation variables indicate whether a certain connection is used
                if relax:
                    y[i, j] = m.addVar(
                        obj=fix_costs[i, j], vtype=gp.GRB.CONTINUOUS, name=f"y[{i},{j}]"
                    )
                else:
                    y[i, j] = m.addVar(
                        obj=fix_costs[i, j], vtype=gp.GRB.INTEGER, name=f"y[{i},{j}]"
                    )

    # Set activation variable to one if the connection is used by some flow
    for i, j in x.keys():
        m.addConstr(x[i, j] <= vehicle_capacities[i, j] * y[i, j])

    return m, x, y


##################################################
# Helper functions
##################################################


def sol_vals(var_dict):
    """Translate solution into a dictionary.

    Parameters
    ----------
    var_dict: dict
        Dictionary containing model variables.

    Returns
    -------
    dict
        Dictionary containing variable values.

    """
    return {k: np.round(v.X) for k, v in var_dict.items()}


def get_fctp_bfs(instance, edge_mask=None):
    """Get basic feasible solution for (reduced) FCTP.

    Step 1: Solve linear relaxation
    Step 2: Create proper basic feasible solution

    Parameters
    ----------
    instance: FCTP
        FCTP instance to be solved.
    edge_mask: 2D np.array, optional
        Binary adjacency matrix indicating whether connection should be included.

    Returns
    -------
    bfs: dict
        Basic feasible solution {(i,j): flow}.

    """
    lp_costs = get_weight_matrix(
        instance.var_costs,
        instance.fix_costs,
        instance.supply,
        instance.demand,
        only_var_costs=False,
        inverse=False,
    )
    if edge_mask is None:
        model, x = tp(instance.supply, instance.demand, lp_costs)
    else:
        model, x = tp_subset_connections(
            instance.supply, instance.demand, lp_costs, edge_mask
        )
    model.setParam("OutputFlag", 0)
    model.optimize()
    bfs = get_basis(x, len(instance.supply), len(instance.demand))
    return bfs


def get_basis(x, m, n):
    """Get proper basis by filling up solution with zero-edges.

    Note: A BFS for the FCTP contains m + n - 1 edges that form a spanning tree.

    Parameters
    ----------
    x: dict
        Dictionary of flow variables, keyed by (supplier, customer).
    m: int
        Number of suppliers.
    n: int
        Number of customers.

    Returns
    -------
    bfs: dict
        Basic feasible solution {(i,j): flow}.

    """
    bfs = {}
    nbv = []
    for k, v in x.items():
        if v.vbasis == 0:
            bfs[k] = v.X
        else:
            nbv.append(k)

    size = m + n - 1
    if not len(bfs) == size:
        bfs = fill_basis_with_nbv(bfs, nbv, m, n, max_tries=50)

    assert len(bfs) == m + n - 1, "Basis does not have the correct size"
    return bfs


def fill_basis_with_nbv(bfs, nbv, m, n, max_tries=50):
    """Get proper basis by filling up solution with zero-edges.

    Note: A BFS for the FCTP contains m + n - 1 edges that form a spanning tree.

    This functions sequentially adds zero-value edges to a solution until a proper basis is
    obtained.

    Parameters
    ----------
    bfs: dict
        Potential basic feasible solution {(i,j): flow}.
    nbv: list
        List of non-basic variables.
    m: int
        Number of suppliers.
    n: int
        Number of customers.
    max_tries: int, optional
        Maximum number of retries with different sequence of NBVs if first try was unsuccessful.

    Returns
    -------
    bfs: dict
        Basic feasible solution {(i,j): flow}.

    """

    size = m + n - 1

    # make sure that the bfs indeed forms a spanning tree
    spanning_tree_graph = get_spanning_tree(bfs, m, n)
    try:
        nx.find_cycle(spanning_tree_graph)
        raise Exception("Spanning tree contains cycles")
    except:
        pass

    for i in range(max_tries):
        # print(f"attempt {i}")
        bfs_tmp = bfs.copy()
        spanning_tree_graph_tmp = spanning_tree_graph.copy()

        # add zero variables into the basis if they do not create a cycle until the target size has been reached
        for k in nbv:
            # tentatively add NBV to basis and check whether a cycle is formed
            spanning_tree_copy = spanning_tree_graph_tmp.copy()
            spanning_tree_copy.add_edge(f"S{k[0]}", f"D{k[1]}")
            try:
                nx.find_cycle(spanning_tree_copy)
                continue
            except:
                bfs_tmp[k] = 0
                spanning_tree_graph_tmp.add_edge(f"S{k[0]}", f"D{k[1]}")
                if len(bfs_tmp) == size:
                    return bfs_tmp

        # add NBVs in a different sequence
        random.shuffle(nbv)
    return bfs


def get_spanning_tree(bfs, m, n):
    """Get Spanning Tree Graph representing BFS.

    Parameters
    ----------
    bfs: dict
        Basic feasible solution {(i,j): flow}.
    m: int
        Number of suppliers.
    n: int
        Number of customers.

    Returns
    -------
    graph: nx.Graph
        Spanning Tree Graph.

    """
    graph = nx.Graph()
    supply_nodes = [f"S{i}" for i in range(m)]
    demand_nodes = [f"D{j}" for j in range(n)]
    edges = [(f"S{i}", f"D{j}") for i, j in bfs.keys()]
    graph.add_nodes_from(supply_nodes)
    graph.add_nodes_from(demand_nodes)
    graph.add_edges_from(edges)
    return graph
