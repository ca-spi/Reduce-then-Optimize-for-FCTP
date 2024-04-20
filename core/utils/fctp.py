"""Classes and functionalities to represent FCTP instances."""

import numpy as np


class FCTP:
    """Class representing FCTP instance.

    Attributes
    ----------
    supply: np.array
        Supply vector.
    demand: np.array
        Demand vector.
    var_costs: np.array
        Variable cost matrix.
    fix_costs: np.array
        Fixed-cost matrix.
    supplier_locations: np.array, optional
        Supplier locations (m x 2).
    customer_locations: np.array, optional
        Customer locations (m x 2).

    """

    def __init__(
        self,
        supply,
        demand,
        var_costs,
        fix_costs,
        supplier_locations=None,
        customer_locations=None,
    ):
        self.supply = supply
        self.demand = demand
        self.var_costs = var_costs
        self.fix_costs = fix_costs
        self.supplier_locations = supplier_locations
        self.customer_locations = customer_locations

    @property
    def m(self):
        """Number of supply nodes."""
        return len(self.supply)

    @property
    def n(self):
        """Number of demand nodes."""
        return len(self.demand)

    def to_dict(self):
        attributes = vars(self)
        attributes["instance_type"] = "fctp"
        return attributes

    def eval_sol_dict(self, solution):
        """Compute total costs for solution provided as dict.

        Parameters
        ----------
        solution: dict
            Solution dictionary containing flows from supplier i to customer j.

        Returns
        -------
        float or int
            Total costs incurred by solution.

        """
        v_costs = sum([v * self.var_costs[i, j] for (i, j), v in solution.items()])
        f_costs = sum([self.fix_costs[i, j] for (i, j), v in solution.items() if v > 0])
        return v_costs + f_costs

    def eval_sol_matrix(self, solution):
        """Compute total costs for solution provided as matrix.

        Parameters
        ----------
        solution: np.array
            Solution matrix (m x n) containing flows from supplier i to customer j.

        Returns
        -------
        float or int
            Total costs incurred by solution.

        """
        return np.sum(
            solution * self.var_costs + np.where(solution > 0, 1, 0) * self.fix_costs,
        )


class CapacitatedFCTP(FCTP):
    """Class representing capacitated FCTP instance.

    Attributes
    ----------
    supply: np.array
        Supply vector.
    demand: np.array
        Demand vector.
    var_costs: np.array
        Variable cost matrix (m x n).
    fix_costs: np.array
        Fixed-cost matrix (m x n).
    edge_capacities: np.array
        Edge capacity matrix (m x n).
    supplier_locations: np.array, optional
        Supplier locations (m x 2).
    customer_locations: np.array, optional
        Customer locations (m x 2).

    """

    def __init__(
        self,
        supply,
        demand,
        var_costs,
        fix_costs,
        edge_capacities,
        supplier_locations=None,
        customer_locations=None,
    ):
        super().__init__(
            supply, demand, var_costs, fix_costs, supplier_locations, customer_locations
        )

        self.edge_capacities = edge_capacities

    def to_dict(self):
        attributes = vars(self)
        attributes["instance_type"] = "c-fctp"
        return attributes


class FixedStepFCTP(FCTP):
    """Class representing capacitated FCTP instance with fixed-step costs.

    Attributes
    ----------
    supply: np.array
        Supply vector.
    demand: np.array
        Demand vector.
    var_costs: np.array
        Variable cost matrix (m x n).
    fix_costs: np.array
        Fixed-cost matrix (m x n).
    vehicle_capacities: np.array
        Vehicle capacity matrix (m x n).
    supplier_locations: np.array, optional
        Supplier locations (m x 2).
    customer_locations: np.array, optional
        Customer locations (m x 2).

    """

    def __init__(
        self,
        supply,
        demand,
        var_costs,
        fix_costs,
        vehicle_capacities,
        supplier_locations=None,
        customer_locations=None,
    ):
        super().__init__(
            supply, demand, var_costs, fix_costs, supplier_locations, customer_locations
        )

        self.vehicle_capacities = vehicle_capacities

    def to_dict(self):
        attributes = vars(self)
        attributes["instance_type"] = "fs-fctp"
        return attributes

    def eval_sol_dict(self, solution):
        """Compute total costs for solution provided as dict.

        Parameters
        ----------
        solution: dict
            Solution dictionary containing flows from supplier i to customer j.

        Returns
        -------
        float or int
            Total costs incurred by solution.

        """
        v_costs = sum([v * self.var_costs[i, j] for (i, j), v in solution.items()])
        f_costs = sum(
            [
                self.fix_costs[i, j] * np.ceil(v / self.vehicle_capacities[i, j])
                for (i, j), v in solution.items()
                if v > 0
            ]
        )
        return v_costs + f_costs

    def eval_sol_matrix(self, solution):
        """Compute total costs for solution provided as matrix.

        Parameters
        ----------
        solution: np.array
            Solution matrix (m x n) containing flows from supplier i to customer j.

        Returns
        -------
        float or int
            Total costs incurred by solution.

        """
        v_costs = solution * self.var_costs
        f_costs = np.ceil(solution / self.vehicle_capacities) * self.fix_costs
        return np.sum(v_costs + f_costs)
