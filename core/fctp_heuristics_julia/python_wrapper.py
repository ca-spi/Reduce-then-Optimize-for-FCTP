"""Python wrapper for Julia implementations of FCTP meta-heuristics. """

import os

import numpy as np

from core.utils.utils import get_random_string
from core.utils.postprocessing import matrix_to_dict


class TabuSearchJuliaEnv:
    """Wrapper for Julia implementation of Tabu Search."""

    def __init__(self):
        from julia.api import Julia

        jl = Julia(compiled_modules=False)
        jl.eval('include("core/fctp_heuristics_julia/sun_ts.jl")')
        self.ts = jl.eval("TabuSearch.ts_wrapper")
        self.compile_run()

    def compile_run(self):
        """Perform compiliation run."""
        supply = np.array([4, 2, 3])
        demand = np.array([4, 3, 2])
        var_costs = np.array([[3.0, 2.0, 4.0], [5.0, 2.0, 2.0], [3.0, 4.0, 3.0]])
        fix_costs = np.array(
            [[40.0, 35.0, 30.0], [25.0, 45.0, 30.0], [35.0, 35.0, 40.0]]
        )
        edge_mask = [[True, False, True], [True, True, False], [False, True, True]]
        bfs = {(1, 1): 4, (2, 1): 0, (2, 2): 2, (3, 2): 1, (3, 3): 2}
        test_log_file = f"{get_random_string(10)}_test.txt"
        config = {
            "tabu_in_range": (7, 10),
            "tabu_out_range": (2, 4),
            "beta": 0.5,
            "gamma": 0.5,
            "L": 3,
            "seed": 0,
            "log_file": test_log_file,
        }
        self.ts(supply, demand, var_costs, fix_costs, edge_mask, bfs, config)
        try:
            os.remove(test_log_file)
        except OSError:
            pass

    def run(self, instance, bfs, config, edge_mask=None):
        """Run Tabu Search to solve FCTP.

        Parameters
        ----------
        instance: FCTP
            Instance to solve.
        bfs: dict
            Basic feasible solution (start solution).
        config: dict
            Tabu Search configuration.
        edge_mask: 2D np.array, optional
            Boolean edge mask indicating relevant edges.

        Returns
        -------
        dict:
            Solution dictionary.
        sol_val: float
            Objective function value.
        runtime: float
            Runtime.

        """
        if edge_mask is None:
            edge_mask = np.full(instance.var_costs.shape, True, dtype=bool)
        bfs = {(i + 1, j + 1): int(v) for (i, j), v in bfs.items()}
        sol, sol_val, runtime = self.ts(
            instance.supply.astype(int),
            instance.demand.astype(int),
            instance.var_costs.astype("float64"),
            instance.fix_costs.astype("float64"),
            edge_mask,
            bfs,
            config,
        )
        return matrix_to_dict(sol, False), sol_val, runtime


class EvolutionaryAlgorithmJuliaEnv:
    """Wrapper for Julia implementation of Evolutionary Algorithm."""

    def __init__(self):
        from julia.api import Julia

        # Load julia function
        jl = Julia(compiled_modules=False)
        jl.eval('include("core/fctp_heuristics_julia/eckert_ea.jl")')
        self.ea = jl.eval("EvolutionaryAlgorithm.ea_wrapper")
        self.compile_run()

    def compile_run(self):
        """Perform compiliation run."""
        supply = np.array([4, 2, 3])
        demand = np.array([4, 3, 2])
        var_costs = np.array([[3.0, 2.0, 4.0], [5.0, 2.0, 2.0], [3.0, 4.0, 3.0]])
        fix_costs = np.array(
            [[40.0, 35.0, 30.0], [25.0, 45.0, 30.0], [35.0, 35.0, 40.0]]
        )
        edge_mask = [[False, True, True], [True, True, False], [True, True, True]]
        for mutation_operator in ["eicr", "nlo"]:
            test_log_file = f"{get_random_string(10)}_test.txt"
            config = {
                "pop_size": 3,
                "max_unique_sols": 10,
                "patience": 100,
                "mutation_operator": mutation_operator,
                "seed": 0,
                "log_file": test_log_file,
            }
            self.ea(supply, demand, var_costs, fix_costs, edge_mask, config)
            try:
                os.remove(test_log_file)
            except OSError:
                pass

    def run(self, instance, config, edge_mask=None):
        """Run EA to solve FCTP.

        Parameters
        ----------
        instance: FCTP
            FCTP instance to solve.
        config: dict
            EA configuration.
        edge_mask: 2D np.array, optional
            Boolean edge mask indicating relevant edges.

        Returns
        -------
        dict:
            Solution dictionary.
        sol_val: float
            Objective function value.
        runtime: float
            Runtime.

        """
        if edge_mask is None:
            edge_mask = np.full(instance.var_costs.shape, True, dtype=bool)
        sol, sol_val, runtime = self.ea(
            instance.supply,
            instance.demand,
            instance.var_costs.astype("float64"),
            instance.fix_costs.astype("float64"),
            edge_mask,
            config,
        )
        return matrix_to_dict(sol, False), sol_val, runtime
