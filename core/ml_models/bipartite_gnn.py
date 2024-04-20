""" Pytorch implemenation of bipartite GNN."""

import torch
import torch.nn.functional as F


class GraphLayerAtt(torch.nn.Module):
    """Graph convolutional layer with attention.

    Parameters
    ----------
    dims_in: tuple
        3-element tuple containing input dimension for supply nodes, demand nodes, and edges,
        respectively.
    dims_out: tuple
        3-element tuple containing output dimension for supply nodes, demand nodes, and edges,
        respectively.
    weight_init_func: function, optional
        Weight initialization function.

    """

    def __init__(self, dims_in, dims_out, weight_init_func=None):
        super(GraphLayerAtt, self).__init__()

        in_s, in_d, in_e = dims_in
        out_s, out_d, out_e = dims_out

        self.dense_ss = torch.nn.Linear(in_s, out_s)
        self.dense_se = torch.nn.Linear(in_e, out_s, bias=False)
        self.attention_se = torch.nn.Linear(in_e, 1)

        self.dense_dd = torch.nn.Linear(in_d, out_d)
        self.dense_de = torch.nn.Linear(in_e, out_d, bias=False)
        self.attention_de = torch.nn.Linear(in_e, 1)

        self.dense_ee = torch.nn.Linear(in_e, out_e)
        self.dense_es = torch.nn.Linear(in_s, out_e, bias=False)
        self.dense_ed = torch.nn.Linear(in_d, out_e, bias=False)

        if weight_init_func is not None:
            weight_init_func(self.dense_ss.weight)
            weight_init_func(self.dense_se.weight)
            weight_init_func(self.dense_dd.weight)
            weight_init_func(self.dense_de.weight)
            weight_init_func(self.dense_ee.weight)
            weight_init_func(self.dense_es.weight)
            weight_init_func(self.dense_ed.weight)

    def forward(self, x_s, x_d, x_e):
        # x_s: b x n x 1, x_d: b x m x 1, x_e: b x n x m x 2
        h_s = self.dense_ss(x_s) + torch.sum(
            self.dense_se(x_e) * self.attention_se(x_e).softmax(dim=2), dim=2
        )

        h_d = self.dense_dd(x_d) + torch.sum(
            self.dense_de(x_e) * self.attention_de(x_e).softmax(dim=1), dim=1
        )
        h_e = (
            self.dense_ee(x_e)
            + self.dense_es(x_s)[:, :, None, :]
            + self.dense_ed(x_d)[:, None, :, :]
        )
        return h_s, h_d, h_e


class GraphNNAtt(torch.nn.Module):
    """GNN with attention.

    Parameters
    ----------
    dims_in: tuple
        3-element tuple containing input dimension for supply nodes, demand nodes, and edges,
        respectively.
    conv_dims: tuple
        Output dimensions of graph convolutional layers. The length of the tuple defines the
        number of convolutional layers.
    dense_dims: tuple
        Output dimensions of dense layers. The length of the tuple defines the number of dense
        layers.
    dim_out: int
        Edge output dimension.
    weight_init_func: function, optional
        Weight initialization function.

    """

    def __init__(
        self,
        dims_in,
        conv_dims,
        dense_dims,
        dim_out,
        weight_init_func=None,
    ):
        super(GraphNNAtt, self).__init__()

        # convolution layers
        self.conv = torch.nn.ModuleList()
        self.conv.append(
            GraphLayerAtt(dims_in, conv_dims[0], weight_init_func=weight_init_func)
        )
        for i in range(1, len(conv_dims)):
            self.conv.append(
                GraphLayerAtt(
                    conv_dims[i - 1], conv_dims[i], weight_init_func=weight_init_func
                )
            )

        # dense layers
        self.dense = torch.nn.ModuleList()
        if len(dense_dims) >= 1:
            self.dense.append(torch.nn.Linear(conv_dims[-1][-1], dense_dims[0]))
            for i in range(1, len(dense_dims)):
                self.dense.append(torch.nn.Linear(dense_dims[i - 1], dense_dims[i]))

        # output layer
        self.output_dim = dim_out
        if len(dense_dims) >= 1:
            self.out = torch.nn.Linear(dense_dims[-1], dim_out)
        else:
            self.out = torch.nn.Linear(conv_dims[-1][-1], dim_out)

    def forward(self, x_s, x_d, x_e):
        for conv_layer in self.conv:
            x_s, x_d, x_e = conv_layer(x_s, x_d, x_e)
            x_s, x_d, x_e = F.relu(x_s), F.relu(x_d), F.relu(x_e)

        for dense_layer in self.dense:
            x_e = F.relu(dense_layer(x_e))

        output = self.out(x_e)

        return output
