import torch
import torch.nn as nn
from typing import Sequence
from itertools import chain


class MLP(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: Sequence[int],
            output_dim: int
    ):
        super().__init__()
        dimension_chain = list(chain([input_dim], hidden_dim, [output_dim]))

        self.layers = nn.ModuleList(
            [nn.Linear(dimension_chain[i], dimension_chain[i + 1]) for i in range(len(dimension_chain) - 1)]
        )
        self.relu = nn.ReLU()
        self.input_dim=input_dim
        self.output_dim=output_dim

    def forward(self, x):
        for i, l in enumerate(self.layers):
            x = l(x)

            if i != len(self.layers) - 1:
                x = self.relu(x)

        return x
