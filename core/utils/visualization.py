""" Functions for visualizing FCTP instances and results. """

import itertools
import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import plotly.express as px

from core.utils.postprocessing import matrix_to_dict


def warmstart_px_save(path):
    """Warmstart plotly.express PDF-saving.

    Parameters
    ----------
    path: str
        Path to save figure.

    Returns
    -------
    None

    """
    fig = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])
    fig.write_image(path, format="pdf")
    time.sleep(1)


def random_hex_color():
    """Compute a random hex color code.

    Returns
    -------
    str
        Random hex color code.
    """
    r = lambda: np.random.randint(0, 230)
    return "#{:02x}{:02x}{:02x}".format(r(), r(), r())
    # return "#%02X%02X%02X" % (r(), r(), r())


def plot_fctp(
    supplier_locations,
    customer_locations,
    supply=None,
    demand=None,
    with_quantities=False,
    draw_edges=True,
    edges=None,
    edge_widths=None,
    clusters=False,
    supplier_clusters=None,
    customer_clusters=None,
    cluster_colors=None,
    with_labels=True,
    ax=None,
    show=True,
):
    """Basic function for plotting a FCTP instance.

    Parameters
    ----------
    supplier_locations: list
        A list of supplier locations, each specified as a two-element-tuple
        (x-coordinate, y-coordinate).
    customer_locations: list
        A list of customer objects, each specified as a two-element-tuple
        (x-coordinate, y-coordinate).
    draw_edges: bool, optional
        Indicates whether edges should be drawn. Default is True.
    edges: list, optional
        List of edges, provided as (origin, destination)-tuples, that should be drawn. Only
        considered if draw_edges=True. If draw_edges=True and no edges are provided, then edges
        between all suppliers and customers are drawn.
    edge_widths: list, optional
        Contains the desired edge widths for each of the edges.
    clusters: bool, optional
        Indicates whether clusters should be considered. Default is False.
    supplier_clusters: dict, optional
        Specifies for each supplier the cluster he belongs to.
    customer_cluster: dict, optional
        Specifies for customer the cluster he belongs to.
    cluster_colors: dict, optional
        Specifies the color for each cluster. If not provided, the color is derived from the
        cluster number.
    with_labels: bool, optional
        Indicates whether node labels should be printed. Default is True.
    ax: matplotlib.axes.Axes, optional
        Axis to plot on.
    show: bool, optional
        Indicate whether figure should be displayed. Default is True.

    Returns
    -------
    None

    """
    prefix_supply = "s"
    prefix_demand = "c"
    # convert supplier and demand information into a graph
    G = nx.DiGraph()
    # add supplier nodes, potentially taking clusters into account
    G.add_nodes_from(
        [
            (
                "{}{}".format(prefix_supply, i),
                {
                    "x": supplier[0],
                    "y": supplier[1],
                    "q": supply[i] if supply is not None else None,
                    "color": (
                        cluster_colors[supplier_clusters[i]]
                        if clusters and cluster_colors is not None
                        else supplier_clusters[i] if clusters else "#73A061"
                    ),
                },
            )
            for i, supplier in enumerate(supplier_locations)
        ]
    )
    # add customer node, potentially taking clusters into account
    G.add_nodes_from(
        [
            (
                "{}{}".format(prefix_demand, i),
                {
                    "x": customer[0],
                    "y": customer[1],
                    "q": demand[i] if demand is not None else None,
                    "color": (
                        cluster_colors[customer_clusters[i]]
                        if clusters and cluster_colors is not None
                        else customer_clusters[i] if clusters else "#64BACD"
                    ),
                },
            )
            for i, customer in enumerate(customer_locations)
        ]
    )
    # optional: add edges
    if draw_edges:
        if edges:
            G.add_edges_from(
                [
                    (
                        "{}{}".format(prefix_supply, i),
                        "{}{}".format(prefix_demand, j),
                    )
                    for i, j in edges
                ]
            )
        else:
            connections = itertools.product(
                range(len(supplier_locations)), range(len(customer_locations))
            )
            G.add_edges_from(
                [
                    (
                        "{}{}".format(prefix_supply, i),
                        "{}{}".format(prefix_demand, j),
                    )
                    for i, j in connections
                ]
            )

    # draw graph
    pos = {
        node_name: (node_attr["x"], node_attr["y"])
        for node_name, node_attr in list(G.nodes(data=True))
    }
    node_colors = [node_attr["color"] for _, node_attr in list(G.nodes(data=True))]
    if with_quantities and all(
        node_attr["q"] for _, node_attr in list(G.nodes(data=True))
    ):
        labels = {
            node: "{}({})".format(node, node_attr["q"])
            for node, node_attr in list(G.nodes(data=True))
        }
    else:
        labels = None
    edge_color = "#797878"
    nx.draw_networkx(
        G,
        pos,
        node_color=node_colors,
        edge_color=edge_color,
        with_labels=with_labels,
        labels=labels,
        node_size=100,
        width=edge_widths if edge_widths is not None else 1,
        ax=ax,
    )

    # show plot
    if show:
        plt.show()


def plot_solution(
    supplier_locations,
    customer_locations,
    sol,
    weight_by_value=False,
    ax=None,
    show=True,
):
    """Wrapper function to plot FCTP instance and solution.

    Parameters
    ----------
    supplier_locations: list
        A list of supplier locations, each specified as a two-element-tuple
        (x-coordinate, y-coordinate).
    customer_locations: list
        A list of customer objects, each specified as a two-element-tuple
        (x-coordinate, y-coordinate).
    sol: dict
        Solution dictionary containing flows from supplier i to customer j.
    weight_by_value: bool, optional
        Indicate whether edge widths should depend on flow quantity. Default is False.
    ax: matplotlib.axes.Axes, optional
        Axis to plot on.
    show: bool, optional
        Indicate whether figure should be displayed. Default is True.

    Returns
    -------
    None

    """
    # identify connections with non-zero flows -> edges that should be included in plot
    sol_edges = [(i, j) for (i, j), v in sol.items() if v > 0]
    # optional: compute edge widths based on flow quantities
    edge_widths = None
    if weight_by_value:
        edge_widths = np.array([sol[edge] for edge in sol_edges])
        edge_widths = list((edge_widths / np.max(edge_widths)) * 3)
    # plot FCTP instance and solution
    plot_fctp(
        supplier_locations,
        customer_locations,
        draw_edges=True,
        edges=sol_edges,
        edge_widths=edge_widths,
        with_labels=True,
        ax=ax,
        show=show,
    )


def visual_evaluation_euclidian_sol_edge_prediction(
    supplier_locations,
    customer_locations,
    matrices,
    title=None,
    subtitles=None,
    text=None,
):
    """Plot Euclidian FCTP solution graphs to illustrate performance.

    Parameters
    ----------
    supplier_locations: list
        A list of supplier locations, each specified as a two-element-tuple
        (x-coordinate, y-coordinate).
    customer_locations: list
        A list of customer objects, each specified as a two-element-tuple
        (x-coordinate, y-coordinate).
    matrices: list
        List containing edge matrices.
    title: str, optional
        Plot title.
    subtitles: list
        List containing titles for each sub-plot.
    text: str, optional
        Additional text to print at bottom of plot.

    Returns
    -------
    fig: matplotlib.figure.Figure
        Figure containing plots.
    ax: matplotlib.axes.Axes
        Axes object.

    """
    matrices = list(matrices)

    n = len(matrices)
    h = 4
    w = n * h
    fig, ax = plt.subplots(ncols=n, figsize=(w, h))

    for i, sol in enumerate(matrices):
        plot_solution(
            supplier_locations,
            customer_locations,
            sol=matrix_to_dict(sol, include_zeros=False),
            weight_by_value=i == 1,
            ax=ax[i],
            show=False,
        )
        if subtitles is not None:
            ax[i].set_title(subtitles[i], fontsize=20)

    if title is not None:
        fig.suptitle(title, fontsize=28)

    if text is not None:
        fig.text(0.5, 0.05, text, horizontalalignment="center", fontsize=20)
        fig.tight_layout(rect=[0, 0.1, 1, 1])
    else:
        fig.tight_layout()

    return fig, ax
