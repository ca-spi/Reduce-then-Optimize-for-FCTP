""" Functions for generating FCTP instances. """

from copy import copy

import numpy as np
from scipy.spatial import distance_matrix

from core.utils.preprocessing import min_of_supply_demand
from core.utils.fctp import FCTP
from core.utils.fctp import CapacitatedFCTP
from core.utils.fctp import FixedStepFCTP


def generate_random_fctp_instance(
    size=15,
    max_quantity=20,
    cost_structure="agarwal-aneja",
    theta=0.2,
    balance=True,
    supply_demand_ratio=1.0,
):
    """Generate random instance according to distributions of Agarwal and Aneja
    (2012) and Roberti (2015).

    Parameters
    ----------
    size: int, optional
        Number of supply and demand nodes. Default is 15.
    max_quantity: int, optional
        Maximum supply/demand quantity. Default is 20.
    cost_structure: str, optional
        Cost structure that should be assumed. Can be 'agarwal-aneja',
        'roberti', or 'euclidean'. Default is 'agarwal-aneja'.
    theta: float, optional
        Variable-fixed-cost ratio factor. Default is 0.2.
    balance: bool, optional
        Indicate whether instance should be belanced to have a certain
        supply-demand-ratio. Default is True.
    supply_demand_ratio: float, optional
        Supply-demand-ratio. Default is 1.0 (=balenced).

    Returns
    -------
    FCTP
        FCTP instance.

    """
    n_suppliers = size
    n_customers = size

    # supply: interval [1,B] with uniform distribution (paper: B=20)
    lb_s = 1
    ub_s = max_quantity
    supply = np.random.randint(lb_s, ub_s + 1, n_suppliers)

    # demand: interval [1,B] with uniform distribution (paper: B=20)
    lb_d = 1
    ub_d = max_quantity
    demand = np.random.randint(lb_d, ub_d + 1, n_customers)
    # adjust demand quantities to achieve supply-demand-ratio
    if balance:
        total_demand = np.round(supply.sum() / supply_demand_ratio).astype(int)
        demand = balance_quantity(demand, total_demand, min_val=lb_d, max_val=ub_d)

    # cost matrices
    assert cost_structure in [
        "agarwal-aneja",
        "roberti",
        "euclidean",
    ]

    lb_c = 200
    ub_c = 800
    D = demand.sum()
    if cost_structure == "euclidean":
        grid_size = int(np.round(1.2 * (ub_c - lb_c)))
        supplier_locations = np.random.randint(0, grid_size, (n_suppliers, 2))
        customer_locations = np.random.randint(0, grid_size, (n_customers, 2))
        fix_costs = np.round(
            distance_matrix(supplier_locations, customer_locations)
        ).astype(int)
        var_costs = ((fix_costs * theta * (n_suppliers + n_customers - 1)) / D).astype(
            int
        )
    else:
        # costs: interval [200,800] with uniform distribution; unit costs scaled such that
        # certain cost ratio is obtained
        supplier_locations = np.array([])
        customer_locations = np.array([])
        fix_costs = np.random.randint(lb_c, ub_c + 1, (n_suppliers, n_customers))
        if cost_structure == "agarwal-aneja":
            # variable and fixed cost sampled independently
            var_costs = np.random.randint(lb_c, ub_c + 1, (n_suppliers, n_customers))
            scaling_factor = (theta * (n_suppliers + n_customers - 1)) / (
                D * np.mean(var_costs / fix_costs)
            )
            var_costs = np.round(var_costs * scaling_factor).astype(int)
        else:
            # variable cost proportioanl to fix cost
            var_costs = (
                (fix_costs * theta * (n_suppliers + n_customers - 1)) / D
            ).astype(int)

    return FCTP(
        supply=supply,
        demand=demand,
        var_costs=var_costs,
        fix_costs=fix_costs,
        supplier_locations=supplier_locations,
        customer_locations=customer_locations,
    )


def generate_random_capacitated_fctp_instance(
    size=15,
    max_quantity=20,
    cost_structure="agarwal-aneja",
    theta=0.2,
    balance=True,
    supply_demand_ratio=1.0,
    capacity_range=(50, 150),
):
    """Generate random instance with edge capacities.

    Parameters
    ----------
    size: int, optional
        Number of supply and demand nodes. Default is 15.
    max_quantity: int, optional
        Maximum supply/demand quantity. Default is 20.
    cost_structure: str, optional
        Cost structure that should be assumed. Can be 'agarwal-aneja',
        'roberti', or 'euclidean'. Default is 'agarwal-aneja'.
    theta: float, optional
        Variable-fixed-cost ratio factor. Default is 0.2.
    balance: bool, optional
        Indicate whether instance should be belanced to have a certain
        supply-demand-ratio. Default is True.
    supply_demand_ratio: float, optional
        Supply-demand-ratio. Default is 1.0 (=balenced).
    capacity_range: tuple of float, optional
        Range to sample relative edge capacity from (in %). Default is [50,150].

    Returns
    -------
    CapacitatedFCTP
        Capacitated FCTP instance.

    """
    fctp_instance = generate_random_fctp_instance(
        size=size,
        max_quantity=max_quantity,
        cost_structure=cost_structure,
        theta=theta,
        balance=balance,
        supply_demand_ratio=supply_demand_ratio,
    )
    # Sample edge capacities
    max_quantity = min_of_supply_demand(fctp_instance.supply, fctp_instance.demand)
    edge_capacities = np.round(
        max_quantity
        * (
            np.random.randint(
                capacity_range[0], capacity_range[1] + 1, fctp_instance.var_costs.shape
            )
            / 100
        )
    ).astype(int)
    return CapacitatedFCTP(
        supply=fctp_instance.supply,
        demand=fctp_instance.demand,
        var_costs=fctp_instance.var_costs,
        fix_costs=fctp_instance.fix_costs,
        edge_capacities=edge_capacities,
        supplier_locations=fctp_instance.supplier_locations,
        customer_locations=fctp_instance.customer_locations,
    )


def generate_random_fixedstep_fctp_instance(
    size=15,
    max_quantity=20,
    cost_structure="agarwal-aneja",
    theta=0.2,
    balance=True,
    supply_demand_ratio=1.0,
):
    """Generate random instance with vehicle capacities resulting in fixed-step costs.

    Parameters
    ----------
    size: int, optional
        Number of supply and demand nodes. Default is 15.
    max_quantity: int, optional
        Maximum supply/demand quantity. Default is 20.
    cost_structure: str, optional
        Cost structure that should be assumed. Can be 'agarwal-aneja',
        'roberti', or 'euclidean'. Default is 'agarwal-aneja'.
    theta: float, optional
        Variable-fixed-cost ratio factor. Default is 0.2.
    balance: bool, optional
        Indicate whether instance should be belanced to have a certain
        supply-demand-ratio. Default is True.
    supply_demand_ratio: float, optional
        Supply-demand-ratio. Default is 1.0 (=balenced).

    Returns
    -------
    FixedStepFCTP
        Fixed-Step FCTP instance.

    """
    fctp_instance = generate_random_fctp_instance(
        size=size,
        max_quantity=max_quantity,
        cost_structure=cost_structure,
        theta=theta,
        balance=balance,
        supply_demand_ratio=supply_demand_ratio,
    )
    # Sample vehicle capacities
    vehicle_capacities = np.random.randint(
        1, int(max_quantity / 2) + 1, (fctp_instance.m, fctp_instance.n)
    )
    return FixedStepFCTP(
        supply=fctp_instance.supply,
        demand=fctp_instance.demand,
        var_costs=fctp_instance.var_costs,
        fix_costs=np.maximum(
            np.round(fctp_instance.fix_costs / 2).astype(int), 1
        ),  # reduce fixed-costs
        vehicle_capacities=vehicle_capacities,
        supplier_locations=fctp_instance.supplier_locations,
        customer_locations=fctp_instance.customer_locations,
    )


def balance_quantity(quantities, total_quantity, min_val=1, max_val=None):
    """Uniformly adjust quantities to obtain target total quantity.

    Parameters
    ----------
    quantities: array-like
        Original quantities.
    total_quantity: int
        Target quantity.
    min_val: int, optional
        Minimum value for each element. Default is 1.
    max_val: int, optional
        Maximum value for each element.

    Returns
    -------
    quantities: array-like
        Adjusted quantities.

    """
    quantities = copy(quantities)
    n = len(quantities)
    if max_val is None:
        max_val = total_quantity
    # calculate data between target and actual total quantity
    delta = total_quantity - np.sum(quantities)
    if delta == 0:
        return quantities
    # increase or decrease quantities until target is reached
    eps = 1 if delta > 0 else -1
    i = 0
    while delta != 0:
        if min_val <= quantities[i] + eps <= max_val:
            quantities[i] += eps
            delta -= eps
        i += 1
        if i >= n:
            i = 0
    return quantities
