"""Training and evaluation utilities for HAD-GCN."""

from __future__ import annotations

import copy
import csv
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score
from torch import Tensor, nn
from torch.utils.data import DataLoader

LOGGER = logging.getLogger(__name__)


def _move_batch(
    batch: Dict[str, Tensor],
    device: torch.device,
) -> Dict[str, Tensor]:
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
    }


def _forward(
    model: nn.Module,
    batch: Dict[str, Tensor],
) -> Tensor:
    return model(
        raw_eeg=batch["raw_eeg"].float(),
        cwt_image=batch["cwt_image"].float(),
        graph_node_features=batch[
            "graph_node_features"
        ].float(),
        graph_adjacency=batch["graph_adjacency"].float(),
        tf_branch_selection=batch["branch_selection"],
        ecg_adaptive_features=None,
    )


def _metrics(
    loss_sum: float,
    sample_count: int,
    labels: List[np.ndarray],
    predictions: List[np.ndarray],
) -> Dict[str, float]:
    if sample_count == 0:
        raise RuntimeError("DataLoader is empty.")

    y_true = np.concatenate(labels)
    y_pred = np.concatenate(predictions)
    return {
        "loss": loss_sum / sample_count,
        "accuracy": float(np.mean(y_true == y_pred)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    loss_sum = 0.0
    sample_count = 0
    labels = []
    predictions = []

    for batch in loader:
        batch = _move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = _forward(model, batch)
        loss = criterion(logits, batch["label"])
        loss.backward()
        optimizer.step()

        count = batch["label"].size(0)
        loss_sum += float(loss.item()) * count
        sample_count += count
        labels.append(batch["label"].detach().cpu().numpy())
        predictions.append(
            logits.argmax(dim=1).detach().cpu().numpy()
        )

    return _metrics(
        loss_sum,
        sample_count,
        labels,
        predictions,
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    loss_sum = 0.0
    sample_count = 0
    labels = []
    predictions = []

    for batch in loader:
        batch = _move_batch(batch, device)
        logits = _forward(model, batch)
        loss = criterion(logits, batch["label"])

        count = batch["label"].size(0)
        loss_sum += float(loss.item()) * count
        sample_count += count
        labels.append(batch["label"].cpu().numpy())
        predictions.append(
            logits.argmax(dim=1).cpu().numpy()
        )

    return _metrics(
        loss_sum,
        sample_count,
        labels,
        predictions,
    )


@torch.no_grad()
def collect_outputs(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    logits_all = []
    labels_all = []
    cz_all = []

    for batch in loader:
        batch = _move_batch(batch, device)
        logits_all.append(_forward(model, batch).cpu().numpy())
        labels_all.append(batch["label"].cpu().numpy())
        cz_all.append(batch["cz_signal"].float().cpu().numpy())

    return (
        np.concatenate(logits_all),
        np.concatenate(labels_all),
        np.concatenate(cz_all),
    )


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    early_stopping_enabled: bool,
    patience: int,
    log_period: int,
) -> Dict[str, List[float]]:
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_accuracy": [],
        "train_kappa": [],
        "validation_loss": [],
        "validation_accuracy": [],
        "validation_kappa": [],
        "epoch_time_s": [],
    }

    best_loss = float("inf")
    best_state: Optional[Dict[str, Tensor]] = None
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )
        validation_metrics = evaluate(
            model,
            validation_loader,
            criterion,
            device,
        )
        elapsed = time.perf_counter() - start

        for key, value in train_metrics.items():
            history["train_{}".format(key)].append(value)
        for key, value in validation_metrics.items():
            history["validation_{}".format(key)].append(value)
        history["epoch_time_s"].append(elapsed)

        if epoch == 1 or epoch % log_period == 0:
            LOGGER.info(
                "Epoch %03d | train loss %.4f acc %.2f%% kappa %.4f | "
                "validation loss %.4f acc %.2f%% kappa %.4f | %.2fs",
                epoch,
                train_metrics["loss"],
                train_metrics["accuracy"] * 100,
                train_metrics["kappa"],
                validation_metrics["loss"],
                validation_metrics["accuracy"] * 100,
                validation_metrics["kappa"],
                elapsed,
            )

        if validation_metrics["loss"] < best_loss:
            best_loss = validation_metrics["loss"]
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

        if early_stopping_enabled and bad_epochs >= patience:
            LOGGER.info(
                "Early stopping at epoch %d.",
                epoch,
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def save_history(
    history: Dict[str, List[float]],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(history.keys())
    rows = zip(*(history[key] for key in keys))
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(keys)
        writer.writerows(rows)
