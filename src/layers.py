import torch
import torch.nn as nn
from typing import Sequence
from itertools import chain


class MLP(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: Sequence[int],
            positive: bool = False,
            dropout: float = 0.2
    ):
        super().__init__()
        dimension_chain = list(chain([input_dim], hidden_dim))

        self.layers = nn.ModuleList(
            [nn.Linear(dimension_chain[i], dimension_chain[i + 1]) for i in range(len(dimension_chain) - 1)]
        )
        self.act = nn.Sigmoid()
        self.dropout = nn.Dropout(p=dropout)

        self.softplus = nn.Softplus()
        self.input_dim = input_dim
        self.output_dim = hidden_dim[-1]

        self.positive = positive

    def forward(self, x):
        for i, l in enumerate(self.layers):

            x = l(x)

            if i != len(self.layers) - 1:
                x = self.dropout(self.act(x))

        if self.positive:
            x = torch.exp(x)

        return x
