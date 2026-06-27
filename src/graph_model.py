"""Three-layer dense GCN used by the HAD-GCN graph module."""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
from torch import Tensor, nn


def normalize_adjacency(adjacency: Tensor) -> Tensor:
    if adjacency.ndim != 3:
        raise ValueError("Expected adjacency [B, N, N].")
    batch_size, node_count, _ = adjacency.shape
    identity = torch.eye(
        node_count,
        device=adjacency.device,
        dtype=adjacency.dtype,
    ).expand(batch_size, -1, -1)
    adjacency = adjacency + identity
    degree = adjacency.sum(dim=-1).clamp_min(1e-8)
    inverse_sqrt = degree.rsqrt()
    return (
        inverse_sqrt.unsqueeze(-1)
        * adjacency
        * inverse_sqrt.unsqueeze(-2)
    )


class DenseGraphConvolution(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(
            input_dim,
            output_dim,
            bias=False,
        )
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(
        self,
        node_features: Tensor,
        normalized_adjacency: Tensor,
    ) -> Tensor:
        support = self.linear(node_features)
        return (
            torch.bmm(normalized_adjacency, support)
            + self.bias
        )


class GraphLearningModule(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (64, 64, 128),
        output_dim: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if len(hidden_dims) != 3:
            raise ValueError("Exactly three GCN layers are required.")

        dimensions = (input_dim,) + tuple(hidden_dims)
        self.layers = nn.ModuleList(
            [
                DenseGraphConvolution(
                    dimensions[index],
                    dimensions[index + 1],
                )
                for index in range(3)
            ]
        )
        self.norms = nn.ModuleList(
            [nn.LayerNorm(dimension) for dimension in hidden_dims]
        )
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        readout_dim = 2 * sum(hidden_dims)
        self.readout = nn.Sequential(
            nn.Linear(readout_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    @staticmethod
    def _pool(node_features: Tensor) -> Tensor:
        mean = node_features.mean(dim=1)
        maximum = node_features.amax(dim=1)
        return torch.cat((mean, maximum), dim=-1)

    def forward(
        self,
        node_features: Tensor,
        adjacency: Tensor,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        normalized_adjacency = normalize_adjacency(adjacency)
        hidden = node_features
        layer_outputs = []

        for layer, norm in zip(self.layers, self.norms):
            hidden = layer(hidden, normalized_adjacency)
            hidden = self.dropout(
                self.activation(norm(hidden))
            )
            layer_outputs.append(hidden)

        pooled = torch.cat(
            [self._pool(value) for value in layer_outputs],
            dim=-1,
        )
        graph_feature = self.readout(pooled)
        return graph_feature, tuple(layer_outputs)
