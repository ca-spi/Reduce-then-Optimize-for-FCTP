""" Utility functions to evaluate, compare, and plot benchmarking performance. """

from collections import defaultdict
import gzip
import os
import pickle as pkl
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.utils.kpi import shifted_gmean
from core.utils.kpi import stats
from core.utils.kpi import evaluate_solution_costs_dict
from core.utils.utils import default_to_regular_dict
from core.utils.postprocessing import sol_to_matrix

##############################################################
# Benchmarking performance tables for method comparisons
##############################################################


def performance_by_instance(solution_dirs, best_known_vals=None):
    """Report performance of different methods for every instance individually.

    Parameters
    ----------
    solution_dirs: dict
        Dictionary containing paths to benchmarking results {<method_name>: <dir_path>}.
    best_known_vals: list, optional
        Best known objective function value for every instance.

    Returns
    -------
    df: pd.DataFrame
        Performance table with instances as rows and methods and metrics as columns.

    """
    methods = solution_dirs.keys()
    relevant_keys = {
        "solver runtime": "solver_runtime",
        "objective value": "objective_value",
    }
    column_index = pd.MultiIndex.from_product(
        [methods, relevant_keys.keys()], names=["method", "KPI"]
    )
    df = pd.DataFrame(columns=column_index)
    for method, path in solution_dirs.items():
        for instance_file in os.listdir(path):
            instance = [int(s) for s in re.findall(r"\d+", instance_file)][0]
            with gzip.open(os.path.join(path, instance_file), "rb") as f:
                result_dict = pkl.load(f)
            for new_key, orig_key in relevant_keys.items():
                df.at[(instance), (method, new_key)] = result_dict[orig_key]
    df = df.sort_index(axis=0)
    if best_known_vals is None:
        best_known_vals = df.xs("objective value", axis=1, level=1).min(axis=1)
    df["UB"] = best_known_vals

    for instance, row in df.iterrows():
        for method in methods:
            df.at[(instance), (method, "opt gap")] = (
                row[method]["objective value"] / row["UB"].values[0] - 1
            ) * 100

    df = df[["UB"] + list(methods)]
    df = df.drop("objective value", axis=1, level=1)
    return df


def get_runtimes_by_method(solution_dir):
    """Get overall runtimes by method.

    Parameters
    ----------
    solution_dir: str
        Path to directory with benchmarking results.

    Returns
    -------
    runtimes: dict
        Dictionary containing for each method a list of total runtimes for all instances.

    """
    runtimes = defaultdict(lambda: [])
    for root, _, files in os.walk(solution_dir):
        for solution_file in files:
            method = os.path.relpath(root, solution_dir)
            with gzip.open(os.path.join(root, solution_file), "rb") as f:
                result_dict = pkl.load(f)
            runtimes[method].append(result_dict["runtime"])

    return runtimes


def get_solver_runtimes_by_method(solution_dir):
    """Get solver runtimes by method.

    Parameters
    ----------
    solution_dir: str
        Path to directory with benchmarking results.

    Returns
    -------
    solver_runtimes: dict
        Dictionary containing for each method a list of solver runtimes for all instances.

    """
    solver_runtimes = defaultdict(lambda: [])
    for root, _, files in os.walk(solution_dir):
        for solution_file in files:
            method = os.path.relpath(root, solution_dir)
            with gzip.open(os.path.join(root, solution_file), "rb") as f:
                result_dict = pkl.load(f)
            if "solver_runtime" in result_dict:
                solver_runtimes[method].append(result_dict["solver_runtime"])
            else:
                solver_runtimes[method].append(0)

    return solver_runtimes


def get_runtime_limit_hit_by_method(solution_dir):
    """Get number of instances that hit runtime limit by method.

    Parameters
    ----------
    solution_dir: str
        Path to directory with benchmarking results.

    Returns
    -------
    runtime_limit_hit: dict
        Dictionary containing for each method a binary list indicating for each instance whether
        it hit the runtime limit.

    """
    runtime_limit_hit = defaultdict(lambda: [])
    for root, _, files in os.walk(solution_dir):
        for solution_file in files:
            method = os.path.relpath(root, solution_dir)
            with gzip.open(os.path.join(root, solution_file), "rb") as f:
                result_dict = pkl.load(f)
            if "solver_status" in result_dict:
                runtime_limit_hit[method].append(int(result_dict["solver_status"] == 9))
            else:
                runtime_limit_hit[method].append(0)

    return runtime_limit_hit


def get_mip_gap_by_method(solution_dir):
    """Get MIP Gap by method.

    Parameters
    ----------
    solution_dir: str
        Path to directory with benchmarking results.

    Returns
    -------
    mip_gaps: dict
        Dictionary containing for each method a list of MIP gaps for all instances.

    """
    mip_gaps = defaultdict(lambda: [])
    for root, _, files in os.walk(solution_dir):
        for solution_file in files:
            method = os.path.relpath(root, solution_dir)
            with gzip.open(os.path.join(root, solution_file), "rb") as f:
                result_dict = pkl.load(f)
            if "mip_gap" in result_dict:
                mip_gaps[method].append(result_dict["mip_gap"] * 100)
            else:
                mip_gaps[method].append(0)

    return mip_gaps


def get_num_edges_by_method(solution_dir):
    """Get number of selected edges.

    Parameters
    ----------
    solution_dir: str
        Path to directory with benchmarking results.

    Returns
    -------
    num_edges_pred: dict
        Dictionary containing for each method a list stating the number of predicted edges for each
        instance.
    num_edges_enriched: dict
        Dictionary containing for each method a list stating the number of total selected edges for
        each instance.

    """
    num_edges_pred = defaultdict(lambda: [])
    num_edges_enriched = defaultdict(lambda: [])
    for root, _, files in os.walk(solution_dir):
        for solution_file in files:
            method = os.path.relpath(root, solution_dir)
            with gzip.open(os.path.join(root, solution_file), "rb") as f:
                result_dict = pkl.load(f)
            # predicted
            if "num_edges_pred" in result_dict:
                num_edges_pred[method].append(result_dict["num_edges_pred"])
            else:
                num_edges_pred[method].append(0)
            # enriched
            if "num_edges_enriched" in result_dict:
                num_edges_enriched[method].append(result_dict["num_edges_enriched"])
            else:
                num_edges_enriched[method].append(0)

    return num_edges_pred, num_edges_enriched


def get_optgaps_by_method(solution_dir, baseline="exact"):
    """Get optimality gaps by method.

    Parameters
    ----------
    solution_dir: str
        Path to directory with benchmarking results.
    baseline: str, optional
        Method that should be used as baseline for calculating optimality gaps. Default is 'exact'.

    Returns
    -------
    opt_gaps: dict
        Dictionary containing for each method a list of optimality gaps for all instances.

    """
    # Step 1: Compute objective function value based on cost data and solution
    obj_vals = defaultdict(lambda: defaultdict(lambda: 0))
    for root, _, files in os.walk(solution_dir):
        for solution_file in files:
            method = os.path.relpath(root, solution_dir)
            with gzip.open(os.path.join(root, solution_file), "rb") as f:
                result_dict = pkl.load(f)
            # instance_path = result_dict["instance_path"]
            instance_path = os.path.basename(result_dict["instance_path"])
            if "objective_value" in result_dict:
                obj_vals[instance_path][method] = result_dict["objective_value"]
            else:
                with gzip.open(instance_path, "rb") as file:
                    instance = pkl.load(file)
                solution = result_dict["solution"]
                obj_vals[instance_path][method] = evaluate_solution_costs_dict(
                    solution, instance.var_costs, instance.fix_costs
                )
    obj_vals = default_to_regular_dict(obj_vals)

    # Step 2: Compute optimality gaps
    opt_gaps_by_instance = defaultdict(lambda: defaultdict(lambda: 0))
    for instance, instance_performance in obj_vals.items():
        for method, obj_val in instance_performance.items():
            if baseline in instance_performance:
                opt_gaps_by_instance[instance][method] = (
                    obj_val / instance_performance[baseline] - 1
                ) * 100
    opt_gaps_by_instance = default_to_regular_dict(opt_gaps_by_instance)

    # Step 3: Get summary dictionary keyed by method
    opt_gaps = defaultdict(lambda: [])
    for instance, instance_performance in opt_gaps_by_instance.items():
        for method, opt_gap in instance_performance.items():
            opt_gaps[method].append(opt_gap)
    opt_gaps = default_to_regular_dict(opt_gaps)

    return opt_gaps


def get_objval_by_method(solution_dir):
    """Get objective values by method.

    Parameters
    ----------
    solution_dir: str
        Path to directory with benchmarking results.

    Returns
    -------
    obj_vals: dict
        Dictionary containing for each method a list of objective values for all instances.

    """
    obj_vals = defaultdict(lambda: [])
    for root, _, files in os.walk(solution_dir):
        for solution_file in files:
            method = os.path.relpath(root, solution_dir)
            with gzip.open(os.path.join(root, solution_file), "rb") as f:
                result_dict = pkl.load(f)
            # instance_path = result_dict["instance_path"]
            instance_path = os.path.basename(result_dict["instance_path"])
            if "objective_value" in result_dict:
                obj_vals[method].append(result_dict["objective_value"])
            else:
                with gzip.open(instance_path, "rb") as file:
                    instance = pkl.load(file)
                solution = result_dict["solution"]
                obj_vals[method].append(
                    evaluate_solution_costs_dict(
                        solution, instance.var_costs, instance.fix_costs
                    )
                )
    obj_vals = default_to_regular_dict(obj_vals)
    return obj_vals


def get_summary_stats_by_method(data_dict):
    """Get summary statistics (mean, std, etc.) for every method.

    Parameters
    ----------
    data_dict: dict
        Dictionary {<method>: <[v1, v2,...,vn]>} with data.

    Returns
    -------
    summary_stats_dict: dict
        Dictionary with summary statistics {<method>: {'mean':...}}.

    """
    summary_stats_dict = {}
    for method, data in data_dict.items():
        summary_stats_dict[method] = stats(data)
    return summary_stats_dict


def ceil(val, precision=3):
    """Ceil function.

    Parameters
    ----------
    val: float
        Value to round.
    precision: int
        Decimal place to round up to.

    """
    return np.round(val + 0.5 * 10 ** (-precision), precision)


def get_performance_table(solution_dir, baseline="exact", epsilon=1e-4, use_gmean=True):
    """Table reporting performance metrics for every method.

    Rows: methods
    Columns: performance metrics (runtimes, optimality gaps etc.)

    Parameters
    ----------
    solution_dir: str
        Path to directory with benchmarking results.
    baseline: str, optional
        Method that should be used as baseline for calculating optimality gaps. Default is 'exact'.
    epsilon: float, optional
        Epsilon for considering optimality gap a zero. Default is 1e-4.
    use_gmean: bool, optional
        Indicate whether geometric mean should be used. Otherwise, arithmetic mean is used. Default
        is True.

    Returns
    -------
    df: pd.DataFrame
        Performance table.

    """
    ###################################################
    # Retrieve and aggragate performance information
    ###################################################
    num_edges_pred_data, num_edges_enriched_data = get_num_edges_by_method(solution_dir)
    runtime_data = get_runtimes_by_method(solution_dir)
    solver_runtime_data = get_solver_runtimes_by_method(solution_dir)
    mip_gap_data = get_mip_gap_by_method(solution_dir)
    runtime_limit_hit_data = get_runtime_limit_hit_by_method(solution_dir)
    opt_gap_data = get_optgaps_by_method(solution_dir, baseline=baseline)
    objval_data = get_objval_by_method(solution_dir)

    ###################################################
    # Extract relevant information and compute statistics
    ###################################################
    # Compute average number of edges
    num_edges_pred_dict = {
        method: np.mean(data) for method, data in num_edges_pred_data.items()
    }
    num_edges_enriched_dict = {
        method: np.mean(data) for method, data in num_edges_enriched_data.items()
    }
    # Compute average runtimes
    if use_gmean:
        mean_fun = lambda x: shifted_gmean(x, s=10)
    else:
        mean_fun = np.mean
    runtime_dict = {method: mean_fun(data) for method, data in runtime_data.items()}
    solver_runtime_dict = {
        method: mean_fun(data) for method, data in solver_runtime_data.items()
    }
    runtime_limit_hit_dict = {
        method: np.sum(data) for method, data in runtime_limit_hit_data.items()
    }
    mip_gap_dict = {method: np.mean(data) for method, data in mip_gap_data.items()}
    # Compute average objective function value
    objval_dict = {method: np.mean(data) for method, data in objval_data.items()}
    # Compute summary statistics (mean, std,...) about optimality gap for each method
    opt_gap_dict = get_summary_stats_by_method(opt_gap_data)
    # Compute percentage of instances with zero optimality gap
    opt_solved_dict = {
        method: np.mean([int(i < epsilon) for i in data]) * 100
        for method, data in opt_gap_data.items()
    }

    ###################################################
    # Aggregate results in a single performance table
    ###################################################
    performance_dict = {}
    relevant_opt_stats_columns = [
        "mean",
        "median",
        "min",
        "max",
        "std",
    ]
    for model_name, opt_gap_stats in opt_gap_dict.items():
        performance_dict[model_name] = {
            k: opt_gap_stats[k] for k in relevant_opt_stats_columns
        }
        performance_dict[model_name]["#edges predicted"] = num_edges_pred_dict[
            model_name
        ]
        performance_dict[model_name]["#edges used"] = num_edges_enriched_dict[
            model_name
        ]
        performance_dict[model_name]["average runtime"] = runtime_dict[model_name]
        performance_dict[model_name]["average solver runtime"] = solver_runtime_dict[
            model_name
        ]
        performance_dict[model_name]["runtime limit hit"] = runtime_limit_hit_dict[
            model_name
        ]
        performance_dict[model_name]["percentage optimally solved"] = opt_solved_dict[
            model_name
        ]
        performance_dict[model_name]["mip gap"] = mip_gap_dict[model_name]
        performance_dict[model_name]["average obj value"] = objval_dict[model_name]
    df = pd.DataFrame.from_dict(performance_dict, orient="index")

    ###################################################
    # Round values and re-arrange columns
    ###################################################
    for c in ["average runtime", "average solver runtime"]:
        df[c] = df[c].apply(lambda x: ceil(x, 3))
    for c in relevant_opt_stats_columns + [
        "percentage optimally solved",
        "#edges predicted",
        "#edges used",
    ]:
        df[c] = df[c].apply(lambda x: np.round(x, 2))
    df = df[
        [
            "#edges predicted",
            "#edges used",
            "average runtime",
            "average solver runtime",
            "runtime limit hit",
            "percentage optimally solved",
            "mip gap",
            "average obj value",
        ]
        + relevant_opt_stats_columns
    ]

    return df


def get_summary_performance_table(
    solution_dirs,
    relevant_keys=None,
    relevant_methods=None,
    baseline="exact",
    use_gmean=True,
):
    """Summary table reporting performance metrics for every method and data set.

    Rows: methods
    Columns: data sets, performance metrics (runtime, optimality gap, percentage solved optimally)

    Parameters
    ----------
    solution_dirs: dict
        Paths to directories with benchmarking results {<dataset>: <dir_path>}.
    relevant_keys: dict, optional
        Metric/keys to include in summary table {<col_name>: <orig_col_name>}.
    relevant_methods: list, optional
        Methods to include in summary table.
    baseline: str, optional
        Method that should be used as baseline for calculating optimality gaps. Default is 'exact'.
    use_gmean: bool, optional
        Indicate whether geometric mean should be used. Otherwise, arithmetic mean is used. Default
        is True.

    Returns
    -------
    summary_df: pd.DataFrame
        Summary performance table.

    """
    if relevant_keys is None:
        relevant_keys = {
            "runtime": "average solver runtime",
            "opt gap": "mean",
            "opt found": "percentage optimally solved",
        }
    column_index = pd.MultiIndex.from_product(
        [solution_dirs.keys(), relevant_keys.keys()], names=["dataset", "KPI"]
    )
    summary_df = pd.DataFrame(columns=column_index)
    for dataset, solution_dir in solution_dirs.items():
        df = get_performance_table(solution_dir, baseline=baseline, use_gmean=use_gmean)
        for method, row in df.iterrows():
            if relevant_methods is not None and method not in relevant_methods:
                continue
            if not method in summary_df.index:
                summary_df = summary_df.reindex(
                    summary_df.index.values.tolist() + [method]
                )
            for new_key, orig_key in relevant_keys.items():
                summary_df.at[(method), (dataset, new_key)] = row[orig_key]
    return summary_df


##############################################################
# Benchmarking performance across edge set sizes (plots)
##############################################################


def get_performance_table_across_edgesetsizes(
    benchmarking_dir,
    experiment_dir,
    baseline="exact",
    relevant_edgesetsizes=None,
):
    """Get table with performance metrics for different reduction levels.

    Rows: reduction levels
    Columns: performance metrics (runtimes, optimality gaps, etc.)

    Parameters
    ----------
    benchmarking_dir: str
        Path to directory with all benchmarking results, containing a subdirectory for every method.
    experiment_dir: str
        Path to directory with benchmarking results for method of interest, containing
        a subdirectory for every edge set size.
    baseline: str, optional
        Method that should be used as baseline for calculating optimality gaps. Default is 'exact'.
    relevant_edgesetsizes: list, optional
        Edge set sizes to include in summary table.

    Returns
    -------
    df: pd.DataFrame
        Performance table.

    """

    if relevant_edgesetsizes is None:
        relevant_edgesetsizes = [float(i) for i in os.listdir(experiment_dir)]

    df = get_performance_table(benchmarking_dir, baseline=baseline)

    # rename and select rows
    df = df.rename(
        index={
            os.path.relpath(os.path.join(experiment_dir, str(s)), benchmarking_dir): s
            for s in relevant_edgesetsizes
        }
    )
    df = pd.concat(
        [df.loc[[baseline]], df.loc[relevant_edgesetsizes].sort_index(axis=0)]
    )

    return df


def get_performance_plot_across_edgesetsizes(
    benchmarking_dir,
    experiment_dir,
    baseline="exact",
    relevant_edgesetsizes=None,
    title=None,
):
    """Plot performance across reduction levels.

    x-axis: reduction level
    y-axis: performance metric (optimality gap)

    Parameters
    ----------
    benchmarking_dir: str
        Path to directory with all benchmarking results, containing a subdirectory for every method.
    experiment_dir: str
        Path to directory with benchmarking results for method of interest, containing
        a subdirectory for every edge set size.
    baseline: str, optional
        Method that should be used as baseline for calculating optimality gaps. Default is 'exact'.
    relevant_edgesetsizes: list, optional
        Edge set sizes to include in summary table.
    title: str, optional
        Plot title.

    Returns
    -------
    fig: matplotlib.figure.Figure
        Performance plot.

    """
    data = get_performance_table_across_edgesetsizes(
        benchmarking_dir,
        experiment_dir,
        baseline,
        relevant_edgesetsizes,
    )
    # remove baseline entries and select and rename relevant column
    data = data.drop(baseline)
    data = data["mean"].rename("opt gap")

    # create a line plot to plot performance against edge set size
    fig = px.line(data, markers=True, template="simple_white")
    fig.update_layout(
        yaxis_title="optimality gap [%]",
        xaxis_title="subset size [%]",
        xaxis={"tickvals": list(data.index)},
        title="Optimality gap for different edge set sizes" if title is None else title,
        showlegend=False,
        # legend=dict(yanchor="top", y=1, xanchor="right", x=1),
        font_size=14,
    )
    return fig


def get_optgap_vs_runtime_plot(
    benchmarking_dir,
    experiment_dir,
    baseline="exact",
    relevant_edgesetsizes=None,
    title=None,
    xaxis_title="Reduction level",
    yaxis_title=("Optimality gap [%]", "Runtime [s]"),
    xaxis_reversed=False,
    optgap_range=None,
    runtime_range=None,
):
    """Plot performance against runtime across reduction levels.

    x-axis: reduction level
    left y-axis: performance metric (optimality gap)
    right y-axis: runtime

    Parameters
    ----------
    benchmarking_dir: str
        Path to directory with all benchmarking results, containing a subdirectory for every method.
    experiment_dir: str
        Path to directory with benchmarking results for method of interest, containing
        a subdirectory for every edge set size.
    baseline: str, optional
        Method that should be used as baseline for calculating optimality gaps. Default is 'exact'.
    relevant_edgesetsizes: list, optional
        Edge set sizes to include in summary table.
    title: str, optional
        Plot title.
    xaxis_title: str, optional
        Title for x-axis. Default is 'Reduction level'.
    yaxis_title: tuple of str, optional
        Titles for both y-axes. Default is 'Optimality gap [%]' for left y-axis and 'Runtime [s]'
        for right y-axis.
    xaxis_reversed: bool, optional
        Indicate whether x-axis should be reversed (descending values).
    optgap_range: tuple, optional
        Optimality gap value range to be plotted.
    runtime_range: tuple, optional
        Runtime value range to be plotted.

    Returns
    -------
    fig: matplotlib.figure.Figure
        Performance plot.

    """
    data = get_performance_table_across_edgesetsizes(
        benchmarking_dir, experiment_dir, baseline, relevant_edgesetsizes
    )
    # remove baseline entries and select and rename relevant columns
    data = data.drop(baseline)
    data = data[["mean", "median", "average solver runtime"]].rename(
        columns={
            "mean": "mean gap",
            "median": "median gap",
            "average solver runtime": "runtime",
        }
    )

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=data.index.tolist(),
            y=data["mean gap"].tolist(),
            name="mean gap",
            marker_color="blue",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=data.index.tolist(),
            y=data["median gap"].tolist(),
            name="median gap",
            marker_color="orange",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=data.index.tolist(),
            y=data["runtime"].tolist(),
            name="runtime",
            marker_color="green",
        ),
        secondary_y=True,
    )

    fig.update_layout(
        template="simple_white",
        xaxis_title=xaxis_title,
        xaxis={"tickvals": list(data.index), "tickfont": {"size": 16}},
        title=title,
        showlegend=True,
        legend=dict(yanchor="top", y=0.5, xanchor="right", x=0.92),
        font_size=18,
        width=750,
        height=450,
        margin=dict(l=20, r=20, t=30, b=20),
    )
    fig.update_yaxes(title_text=yaxis_title[0], range=optgap_range, secondary_y=False)
    fig.update_yaxes(title_text=yaxis_title[1], range=runtime_range, secondary_y=True)
    if xaxis_reversed:
        fig.update_xaxes(autorange="reversed")

    return fig


def get_optgap_vs_runtime_plot_with_edgeset_barplot(
    benchmarking_dir,
    experiment_dir,
    baseline="exact",
    relevant_edgesetsizes=None,
    title=None,
    xaxis_title="Reduction level",
    yaxis_title=("Optimality gap [%]", "Runtime [s]"),
    xaxis_reversed=False,
    optgap_range=None,
    runtime_range=None,
):
    """Plot performance against runtime across reduction levels + edge set size barplots.

    x-axis: reduction level
    left y-axis: performance metric (optimality gap)
    right y-axis: runtime
    bars: edge set size

    Parameters
    ----------
    benchmarking_dir: str
        Path to directory with all benchmarking results, containing a subdirectory for every method.
    experiment_dir: str
        Path to directory with benchmarking results for method of interest, containing
        a subdirectory for every edge set size.
    baseline: str, optional
        Method that should be used as baseline for calculating optimality gaps. Default is 'exact'.
    relevant_edgesetsizes: list, optional
        Edge set sizes to include in summary table.
    title: str, optional
        Plot title.
    xaxis_title: str, optional
        Title for x-axis. Default is 'Reduction level'.
    yaxis_title: tuple of str, optional
        Titles for both y-axes. Default is 'Optimality gap [%]' for left y-axis and 'Runtime [s]'
        for right y-axis.
    xaxis_reversed: bool, optional
        Indicate whether x-axis should be reversed (descending values).
    optgap_range: tuple, optional
        Optimality gap value range to be plotted.
    runtime_range: tuple, optional
        Runtime value range to be plotted.

    Returns
    -------
    fig: matplotlib.figure.Figure
        Performance plot.

    """
    data = get_performance_table_across_edgesetsizes(
        benchmarking_dir, experiment_dir, baseline, relevant_edgesetsizes
    )
    # remove baseline entries and select and rename relevant columns
    data = data.drop(baseline)
    data = data[
        ["mean", "median", "average solver runtime", "#edges predicted"]
    ].rename(
        columns={
            "mean": "mean gap",
            "median": "median gap",
            "average solver runtime": "runtime",
        }
    )

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=data.index.tolist(),
            y=data["#edges predicted"].tolist(),
            name="#edges",
            text=data["#edges predicted"].round().astype(int),
            # name="#edges [%]",
            # text=np.round((data["#edges predicted"] / 255 * 100), 1),
            textposition="auto",
            marker_color="antiquewhite",  # aliceblue
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=data.index.tolist(),
            y=data["mean gap"].tolist(),
            name="mean gap",
            yaxis="y2",
            marker_color="blue",
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=data.index.tolist(),
            y=data["median gap"].tolist(),
            name="median gap",
            yaxis="y2",
            marker_color="orange",
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=data.index.tolist(),
            y=data["runtime"].tolist(),
            name="runtime",
            yaxis="y3",
            marker_color="green",
        ),
    )

    # Create axis objects
    fig.update_layout(
        xaxis=dict(title=xaxis_title, tickvals=list(data.index), tickfont={"size": 16}),
        yaxis=dict(visible=False),
        yaxis2=dict(
            title=yaxis_title[0],
            anchor="x",
            overlaying="y",
            side="left",
            range=optgap_range,
        ),
        yaxis3=dict(
            title=yaxis_title[1],
            anchor="x",
            overlaying="y",
            side="right",
            range=runtime_range,
        ),
    )
    if xaxis_reversed:
        fig.update_xaxes(autorange="reversed")

    fig.update_layout(
        template="simple_white",
        showlegend=True,
        legend=dict(yanchor="top", y=1.0, xanchor="left", x=0.2),
        font_size=18,
        width=720,
        height=450,
        # margin=dict(l=0, r=0, t=0, b=0),
        title=title,
    )

    return fig


def get_optgap_vs_optsolved_plot(
    benchmarking_dir,
    experiment_dir,
    baseline="exact",
    relevant_edgesetsizes=None,
    title=None,
    xaxis_title=None,
    xaxis_reversed=False,
):
    """Plot optimality gap and #instance solved to optimality across reduction levels.

    x-axis: reduction level
    left y-axis: optimality gap
    right y-axis: #instances solved to optimality

    Parameters
    ----------
    benchmarking_dir: str
        Path to directory with all benchmarking results, containing a subdirectory for every method.
    experiment_dir: str
        Path to directory with benchmarking results for method of interest, containing
        a subdirectory for every edge set size.
    baseline: str, optional
        Method that should be used as baseline for calculating optimality gaps. Default is 'exact'.
    relevant_edgesetsizes: list, optional
        Edge set sizes to include in summary table.
    title: str, optional
        Plot title.
    xaxis_title: str, optional
        Title for x-axis.
    xaxis_reversed: bool, optional
        Indicate whether x-axis should be reversed (descending values).

    Returns
    -------
    fig: matplotlib.figure.Figure
        Performance plot.

    """
    data = get_performance_table_across_edgesetsizes(
        benchmarking_dir, experiment_dir, baseline, relevant_edgesetsizes
    )
    # remove baseline entries and select and rename relevant columns
    data = data.drop(baseline)
    data = data[["mean", "percentage optimally solved"]].rename(
        columns={
            "mean": "mean gap",
        }
    )

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=data.index.tolist(),
            y=data["mean gap"].tolist(),
            name="mean gap",
            marker_color="blue",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=data.index.tolist(),
            y=data["percentage optimally solved"].tolist(),
            name="Optimally solved",
            marker_color="green",
        ),
        secondary_y=True,
    )

    fig.update_layout(
        template="simple_white",
        xaxis_title="reduction level" if xaxis_title is None else xaxis_title,
        xaxis={"tickvals": list(data.index)},
        title=title,
        showlegend=True,
        legend=dict(yanchor="top", y=0.5, xanchor="right", x=0.92),
        font_size=14,
        width=720,
        height=450,
    )
    fig.update_yaxes(title_text="Performance gap [%]", secondary_y=False)
    fig.update_yaxes(title_text="Instances solved to optimality [%]", secondary_y=True)
    if xaxis_reversed:
        fig.update_xaxes(autorange="reversed")

    return fig


def get_optgap_and_mipgap_plot(
    benchmarking_dir,
    experiment_dir,
    baseline="exact",
    relevant_edgesetsizes=None,
    title=None,
    xaxis_title="Reduction level",
    yaxis_title=("Optimality gap [%]", "MIP gap [%]"),
    xaxis_reversed=False,
):
    """Plot optimality gap and MIP gap across reduction levels.

    x-axis: reduction level
    left y-axis: optimality gap
    right y-axis: MIP gap

    Parameters
    ----------
    benchmarking_dir: str
        Path to directory with all benchmarking results, containing a subdirectory for every method.
    experiment_dir: str
        Path to directory with benchmarking results for method of interest, containing
        a subdirectory for every edge set size.
    baseline: str, optional
        Method that should be used as baseline for calculating optimality gaps. Default is 'exact'.
    relevant_edgesetsizes: list, optional
        Edge set sizes to include in summary table.
    title: str, optional
        Plot title.
    xaxis_title: str, optional
        Title for x-axis. Default is 'Reduction level'.
    yaxis_title: tuple of str, optional
        Titles for both y-axes. Default is 'Optimality gap [%]' for left y-axis and 'MIP gap [s]'
        for right y-axis.
    xaxis_reversed: bool, optional
        Indicate whether x-axis should be reversed (descending values).

    Returns
    -------
    fig: matplotlib.figure.Figure
        Performance plot.

    """
    data = get_performance_table_across_edgesetsizes(
        benchmarking_dir, experiment_dir, baseline, relevant_edgesetsizes
    )
    # remove baseline entries and select and rename relevant columns
    data = data.drop(baseline)
    data = data[["mean", "median", "mip gap"]].rename(
        columns={
            "mean": "mean gap",
            "median": "median gap",
            "mip gap": "MIP gap",
        }
    )

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=data.index.tolist(),
            y=data["mean gap"].tolist(),
            name="mean gap",
            marker_color="blue",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=data.index.tolist(),
            y=data["median gap"].tolist(),
            name="median gap",
            marker_color="orange",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=data.index.tolist(),
            y=data["MIP gap"].tolist(),
            name="MIP gap",
            line=dict(color="grey", dash="dash"),
        ),
        secondary_y=True,
    )

    fig.update_layout(
        template="simple_white",
        xaxis_title=xaxis_title,
        xaxis={"tickvals": list(data.index), "tickfont": {"size": 16}},
        title=title,
        showlegend=True,
        legend=dict(yanchor="top", y=0.6, xanchor="right", x=0.92),
        font_size=18,
        width=720,
        height=450,
    )
    fig.update_yaxes(title_text=yaxis_title[0], secondary_y=False)
    fig.update_yaxes(
        title_text=yaxis_title[1],
        range=[0, data["MIP gap"].max() * 1.1],
        secondary_y=True,
    )
    if xaxis_reversed:
        fig.update_xaxes(autorange="reversed")

    return fig


def get_num_edges_pred_vs_used_barplot(
    benchmarking_dir,
    experiment_dir,
    baseline="exact",
    relevant_edgesetsizes=None,
    title=None,
):
    """Compare number of predicted edges to total number of edges.

    Evaluate impact of feasibility edges on total edge set size.

    x-axis: reduction level
    bars: edge set size (predicted and total)

    Parameters
    ----------
    benchmarking_dir: str
        Path to directory with all benchmarking results, containing a subdirectory for every method.
    experiment_dir: str
        Path to directory with benchmarking results for method of interest, containing
        a subdirectory for every edge set size.
    baseline: str, optional
        Method that should be used as baseline for calculating optimality gaps. Default is 'exact'.
    relevant_edgesetsizes: list, optional
        Edge set sizes to include in summary table.
    title: str, optional
        Plot title.

    Returns
    -------
    fig: matplotlib.figure.Figure
        Edge set size plot.

    """

    data = get_performance_table_across_edgesetsizes(
        benchmarking_dir, experiment_dir, baseline, relevant_edgesetsizes
    )
    # remove baseline entries and select relevant columns
    data = data.drop(baseline)
    data = data[["#edges predicted", "#edges used"]]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=data.index.tolist(),
            y=data["#edges used"].tolist(),
            name="Actual edge set size",
            text=data["#edges used"].round().astype(int),
            textposition="outside",
            marker_color="chocolate",  # darkturquoise
        ),
    )

    fig.add_trace(
        go.Bar(
            x=data.index.tolist(),
            y=data["#edges predicted"].tolist(),
            name="Target size",
            text=data["#edges predicted"].round().astype(int),
            textposition="auto",
            marker_color="antiquewhite",  # aliceblue
        ),
    )

    fig.update_layout(
        template="simple_white",
        xaxis=dict(
            # visible=False,
            title="Size threshold",
            tickvals=list(data.index),
        ),
        yaxis=dict(title="#Edges", range=(0, data["#edges used"].max() * 1.1)),
        barmode="overlay",
        showlegend=True,
        legend=dict(yanchor="top", y=1.0, xanchor="left", x=0),
        font_size=14,
        title=title,
        # width=720,
        # height=450,
    )

    return fig


def get_summary_performance_table_across_edgesetsizes(
    benchmarking_dir,
    experiment_dirs,
    baseline="exact",
    metric="mean",
    relevant_edgesetsizes=None,
):
    """Get table with performance (opt gap) for different methods and reduction levels.

    Rows: methods
    Columns: reduction levels

    Parameters
    ----------
    benchmarking_dir: str
        Path to directory with all benchmarking results, containing a subdirectory for every method.
    experiment_dirs: dict
        Path sto directory with benchmarking results for methods of interest {<method>: <dir_path>},
        containing a subdirectory for every edge set size.
    baseline: str, optional
        Method that should be used as baseline for calculating optimality gaps. Default is 'exact'.
    metric: str, optional
        Metric (column) to be used. Default is mean optimality gap ('mean').
    relevant_edgesetsizes: list, optional
        Edge set sizes to include in summary table.

    Returns
    -------
    summary_df: pd.DataFrame
        Summary performance table.

    """

    summary_df = pd.DataFrame(columns=experiment_dirs.keys())
    for method, experiment_dir in experiment_dirs.items():
        method_baseline = baseline[method] if isinstance(baseline, dict) else baseline
        data = get_performance_table_across_edgesetsizes(
            benchmarking_dir,
            experiment_dir,
            baseline=method_baseline,
            relevant_edgesetsizes=relevant_edgesetsizes,
        )
        data = data.drop(method_baseline)
        data = data[metric].rename("opt gap")
        summary_df[method] = data

    return summary_df


def get_summary_performance_plot_across_edgesetsizes(
    benchmarking_dir,
    experiment_dirs,
    baseline="exact",
    metric="mean",
    relevant_edgesetsizes=None,
    title=None,
    xaxis_title="subset size [%]",
    yaxis_title="optimality gap [%]",
    legend_title="Algorithm",
    legend_right=True,
    hline=None,
):
    """Plot performance across reduction levels for different methods.

    x-axis: reduction level
    y-axis: performance metric (mean optimality gap)

    Parameters
    ----------
    benchmarking_dir: str
        Path to directory with all benchmarking results, containing a subdirectory for every method.
    experiment_dirs: dict
        Path sto directory with benchmarking results for methods of interest {<method>: <dir_path>},
        containing a subdirectory for every edge set size.
    baseline: str, optional
        Method that should be used as baseline for calculating optimality gaps. Default is 'exact'.
    metric: str, optional
        Metric (column) to be used. Default is mean optimality gap ('mean').
    relevant_edgesetsizes: list, optional
        Edge set sizes to include in summary table.
    title: str, optional
        Plot title.
    xaxis_title: str, optional
        Title for x-axis. Default is 'subset size [%]'.
    yaxis_title: str, optional
        Title for y-axes. Default is 'optimality gap [%]'.
    legend_title: str, optional
        Legend title. Default is 'Algorithm'.
    legend_right: bool, optional
        Indicate whether legend should be plotted on right-hand side of the plot. Default is True.
    hline: int, optional
        Add baseline line.

    Returns
    -------
    fig: matplotlib.figure.Figure
        Performance plot.

    """

    data = get_summary_performance_table_across_edgesetsizes(
        benchmarking_dir,
        experiment_dirs,
        baseline,
        metric=metric,
        relevant_edgesetsizes=relevant_edgesetsizes,
    )

    # create a line plot to plot performance against edge set size
    fig = px.line(data, markers=True, template="simple_white")
    if hline is not None:
        fig.add_hline(
            y=hline, line_dash="dash", opacity=1, line_width=1, line_color="Black"
        )
    if legend_right:
        legend_dict = dict(yanchor="top", y=1, xanchor="right", x=1)
    else:
        legend_dict = dict(yanchor="top", y=1, xanchor="left", x=0.01)
    fig.update_layout(
        yaxis_title=yaxis_title,
        xaxis_title=xaxis_title,
        xaxis={
            "tickvals": list(data.index),
            "tickangle": 90,
            "tickfont": {"size": 16},
        },
        title=title,
        showlegend=True,
        legend_title=legend_title,
        legend=legend_dict,
        font_size=18,
        margin=dict(l=20, r=20, t=30, b=20),
    )

    return fig


##############################################################
# Other evaluation functions
##############################################################


def proportion_nonzero_edges(solution_dir, absolute_number=False):
    """Calculate average proportion of non-zero edges in solutions.

    Parameters
    ----------
    solution_dir: int
        Path to directory with benchmarking results.
    absolute_number: bool, optional
        Indicate wether number should be absolute (True) or relative to total number of edges.
        Default is False.

    Returns
    -------
    float:
        Average proportion of non-zero edges in solutions.

    """

    prop_nonzeros = []
    for solution_file in os.listdir(solution_dir):
        with gzip.open(os.path.join(solution_dir, solution_file), "rb") as f:
            result_dict = pkl.load(f)
        sol = result_dict["solution"]
        sol = sol_to_matrix(sol)
        if absolute_number:
            prop_nonzeros.append(np.sum(np.where(sol > 0, 1, 0)))
        else:
            prop_nonzeros.append(np.mean(np.where(sol > 0, 1, 0)))
    return np.mean(prop_nonzeros)
