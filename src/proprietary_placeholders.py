"""Patent-sensitive HAD-GCN interfaces.

IMPORTANT
---------
This file intentionally contains NO implementation or approximation of:

1. the time-frequency-domain adaptive branch-selection technique;
2. the ECG adaptive generation technique.

Only variable names, shape validation, and safe runnable fallbacks are
provided. Replace these variables/functions with the protected implementations
after the patent application process is complete.
"""

from __future__ import annotations

from typing import Optional, Union

import torch
from torch import Tensor


# 0 = Raw Signal branch, 1 = CWT branch.
# This is a manual placeholder, not an adaptive classifier.
TF_BRANCH_SELECTION_PLACEHOLDER: Optional[Union[int, Tensor]] = None

# Expected shape: [batch, eeg_nodes, 3]
# The final dimension represents the three supplementary ECG-domain features.
# This is an input variable only; no ECG-generation algorithm is provided.
ECG_ADAPTIVE_FEATURES_PLACEHOLDER: Optional[Tensor] = None


def resolve_tf_branch_selection(
    batch_size: int,
    device: torch.device,
    explicit_selection: Optional[Tensor],
    manual_branch: str,
) -> Tensor:
    """Resolve branch IDs without implementing adaptive selection."""
    selection = (
        explicit_selection
        if explicit_selection is not None
        else TF_BRANCH_SELECTION_PLACEHOLDER
    )

    if selection is None:
        branch_id = 0 if manual_branch == "raw" else 1
        return torch.full(
            (batch_size,),
            branch_id,
            dtype=torch.long,
            device=device,
        )

    if isinstance(selection, int):
        selection = torch.full(
            (batch_size,),
            int(selection),
            dtype=torch.long,
            device=device,
        )
    else:
        selection = torch.as_tensor(
            selection,
            dtype=torch.long,
            device=device,
        ).reshape(-1)

    if selection.numel() == 1:
        selection = selection.expand(batch_size)
    if selection.numel() != batch_size:
        raise ValueError(
            "TF_BRANCH_SELECTION_PLACEHOLDER must contain one value or "
            "one value per batch sample."
        )
    if not torch.all((selection == 0) | (selection == 1)):
        raise ValueError("Branch IDs must be 0 (raw) or 1 (cwt).")
    return selection


def resolve_ecg_adaptive_features(
    batch_size: int,
    node_count: int,
    feature_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    explicit_features: Optional[Tensor],
) -> Tensor:
    """Resolve externally supplied ECG features; default to zeros."""
    features = (
        explicit_features
        if explicit_features is not None
        else ECG_ADAPTIVE_FEATURES_PLACEHOLDER
    )

    if features is None:
        return torch.zeros(
            batch_size,
            node_count,
            feature_dim,
            device=device,
            dtype=dtype,
        )

    features = torch.as_tensor(
        features,
        device=device,
        dtype=dtype,
    )
    expected = (batch_size, node_count, feature_dim)
    if tuple(features.shape) != expected:
        raise ValueError(
            "ECG_ADAPTIVE_FEATURES_PLACEHOLDER must have shape {}, got {}."
            .format(expected, tuple(features.shape))
        )
    return features
