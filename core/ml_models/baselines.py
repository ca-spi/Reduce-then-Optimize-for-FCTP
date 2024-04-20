"""" Pytorch implementation of baseline models. """

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearModel(torch.nn.Module):
    """Linear Model.

    Parameters
    ----------
    input_dim: int
        Input dimensionality.
    output_dim: int
        Output dimensionality.

    """

    def __init__(self, input_dim, output_dim):
        super(LinearModel, self).__init__()
        self.output_dim = output_dim
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs


class MLP(nn.Module):
    """Multi-Layer Perceptron Model.

    Parameters
    ----------
    input_dim: int
        Input dimensionality.
    hidden_dims: array-like
        Dimensionality for each hidden layer. Must not be empty.
    output_dim: int
        Output dimensionality.

    """

    def __init__(self, input_dim, hidden_dims, output_dim):
        super(MLP, self).__init__()

        self.output_dim = output_dim

        self.dense = nn.ModuleList()

        # Input layer
        self.dense.append(nn.Linear(input_dim, hidden_dims[0]))

        # Hidden layers
        for i in range(1, len(hidden_dims)):
            self.dense.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))

        # Output layer
        self.out = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, x):
        for dense_layer in self.dense:
            x = F.relu(dense_layer(x))
        output = self.out(x)
        return output
