""" Custom loss functions. """

import torch


def loss_edges_multiclass(y_pred_edges, y_edges, edge_cw=None):
    """Loss function for multi-class edge predictions.

    Parameters
    ----------
    y_pred_edges: torch.tensor
        Predictions for edges (batch_size x num_supply_nodes x num_demand_nodes x num_classes)
    y_edges: torch.tensor
        Targets for edges (batch_size x num_supply_nodes x num_demand_nodes x num_classes)
    edge_cw: array-like
        Class weights.

    Returns
    -------
    loss_edges:
        Weighted loss.

    """
    y = torch.nn.functional.log_softmax(y_pred_edges, dim=3)  # B x V1 x V2 x VOC
    y = y.permute(0, 3, 1, 2).contiguous()  # B x VOC x V1 x V2
    loss_edges = torch.nn.NLLLoss(edge_cw)(y, y_edges)
    return loss_edges
