"""Trial-graph confidence calibration for HAD-GCN."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score
from torch import Tensor, nn
import torch.nn.functional as F

try:
    from .graph_features import build_trial_correlation_graph
    from .graph_model import DenseGraphConvolution, normalize_adjacency
except ImportError:
    from graph_features import build_trial_correlation_graph
    from graph_model import DenseGraphConvolution, normalize_adjacency


class TrialGraphTemperatureCalibrator(nn.Module):
    """Predict one positive temperature for each trial node."""

    def __init__(
        self,
        n_classes: int,
        hidden_dim: int = 16,
    ) -> None:
        super().__init__()
        self.gcn1 = DenseGraphConvolution(
            n_classes,
            hidden_dim,
        )
        self.gcn2 = DenseGraphConvolution(
            hidden_dim,
            1,
        )

    def forward(
        self,
        logits: Tensor,
        adjacency: Tensor,
    ) -> Tensor:
        if logits.ndim != 2:
            raise ValueError("Expected logits [trials, classes].")
        if adjacency.ndim != 2:
            raise ValueError("Expected adjacency [trials, trials].")

        normalized = normalize_adjacency(
            adjacency.unsqueeze(0)
        )
        hidden = F.relu(
            self.gcn1(
                logits.unsqueeze(0),
                normalized,
            )
        )
        raw_temperature = self.gcn2(
            hidden,
            normalized,
        ).squeeze(0)
        return F.softplus(raw_temperature) + 1e-4


def train_calibrator(
    validation_logits: np.ndarray,
    validation_labels: np.ndarray,
    validation_cz_signals: np.ndarray,
    n_classes: int,
    hidden_dim: int,
    top_k: int,
    epochs: int,
    learning_rate: float,
    device: torch.device,
) -> TrialGraphTemperatureCalibrator:
    adjacency = torch.from_numpy(
        build_trial_correlation_graph(
            validation_cz_signals,
            top_k=top_k,
        )
    ).to(device)
    logits = torch.as_tensor(
        validation_logits,
        dtype=torch.float32,
        device=device,
    )
    labels = torch.as_tensor(
        validation_labels,
        dtype=torch.long,
        device=device,
    )

    model = TrialGraphTemperatureCalibrator(
        n_classes=n_classes,
        hidden_dim=hidden_dim,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        temperature = model(logits, adjacency)
        loss = F.cross_entropy(
            logits / temperature,
            labels,
        )
        loss.backward()
        optimizer.step()

    return model


@torch.no_grad()
def apply_calibrator(
    model: TrialGraphTemperatureCalibrator,
    logits: np.ndarray,
    cz_signals: np.ndarray,
    top_k: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    adjacency = torch.from_numpy(
        build_trial_correlation_graph(
            cz_signals,
            top_k=top_k,
        )
    ).to(device)
    logits_tensor = torch.as_tensor(
        logits,
        dtype=torch.float32,
        device=device,
    )
    temperature = model(
        logits_tensor,
        adjacency,
    )
    calibrated_logits = logits_tensor / temperature
    return (
        calibrated_logits.cpu().numpy(),
        temperature.squeeze(-1).cpu().numpy(),
    )


def calibration_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    confidence_threshold: float = 0.0,
) -> Dict[str, float]:
    logits_tensor = torch.as_tensor(
        logits,
        dtype=torch.float32,
    )
    probabilities = torch.softmax(logits_tensor, dim=1).numpy()
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)

    retained = confidence >= confidence_threshold
    if not np.any(retained):
        return {
            "accuracy": float("nan"),
            "kappa": float("nan"),
            "coverage": 0.0,
            "mean_confidence": float(confidence.mean()),
        }

    return {
        "accuracy": float(
            np.mean(predictions[retained] == labels[retained])
        ),
        "kappa": float(
            cohen_kappa_score(
                labels[retained],
                predictions[retained],
            )
        ),
        "coverage": float(np.mean(retained)),
        "mean_confidence": float(confidence[retained].mean()),
    }
