""" Sample generation. """

from core.utils.fctp import CapacitatedFCTP
from core.utils.fctp import FixedStepFCTP
from core.fctp_solvers.ip_grb import fctp
from core.fctp_solvers.ip_grb import capacitated_fctp
from core.fctp_solvers.ip_grb import fixed_step_fctp
from core.fctp_solvers.ip_grb import sol_vals


def generate_sample(
    instance,
    grb_timeout=60,
    grb_threads=None,
    grb_seed=0,
    verbosity=False,
):
    """Generate sample by solving instance.

    Parameters
    ----------
    instance: FCTP
        Instance to be solved.
    grb_timeout: int, optional
        Gurobi runtime limit in seconds. Default is 60 seconds.
    grb_threads: int, optional
        Number of threads to be used by Gurobi.
    grb_seed: int, optional
        Random seed to be used by Gurobi. Default is 0.
    verbosity: bool, optional
        Indicate whether Gurobi should run in verbose mode.

    Returns
    -------
    dict
        Sample dictionary containing instance, solution, runtime, MIP gap,
        Gurobi status.

    """
    if isinstance(instance, CapacitatedFCTP):
        model, x, _ = capacitated_fctp(
            instance.supply,
            instance.demand,
            instance.var_costs,
            instance.fix_costs,
            instance.edge_capacities,
        )
    elif isinstance(instance, FixedStepFCTP):
        model, x, _ = fixed_step_fctp(
            instance.supply,
            instance.demand,
            instance.var_costs,
            instance.fix_costs,
            instance.vehicle_capacities,
        )
    else:
        model, x, _ = fctp(
            instance.supply, instance.demand, instance.var_costs, instance.fix_costs
        )

    # Solve to optimality
    if not verbosity:
        model.setParam("OutputFlag", 0)
    model.setParam("TimeLimit", grb_timeout)
    model.setParam("Seed", grb_seed)
    if grb_threads is not None:
        model.setParam("Threads", grb_threads)
    model.optimize()
    try:
        sol = sol_vals(x)
    except:
        sol = None

    return {
        "instance": instance,
        "solution": sol,
        "runtime": model.Runtime,
        "opt_gap": model.MIPGap,
        "opt_status": model.Status,
    }
