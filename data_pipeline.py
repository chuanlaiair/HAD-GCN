"""BCI Competition IV-2a data preparation for HAD-GCN."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import mne
import numpy as np
import torch
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt
from torch.utils.data import DataLoader, Dataset

try:
    from .config import ExperimentConfig
    from .cwt_transform import build_tfi_dataset
    from .graph_features import build_pli_adjacencies
except ImportError:
    from config import ExperimentConfig
    from cwt_transform import build_tfi_dataset
    from graph_features import build_pli_adjacencies

LOGGER = logging.getLogger(__name__)

TRIAL_START_CODE = 768
TRAIN_CUE_TO_LABEL = {769: 0, 770: 1, 771: 2, 772: 3}
EVALUATION_CUE_CODE = 783
EOG_CHANNELS = ("EOG-left", "EOG-central", "EOG-right")


@dataclass
class PreparedSession:
    raw_eeg: np.ndarray
    cwt_images: np.ndarray
    graph_node_features: np.ndarray
    graph_adjacency: np.ndarray
    cz_signals: np.ndarray
    labels: np.ndarray
    sampling_rate: float
    channel_names: Tuple[str, ...]


class HADGCNDataset(Dataset):
    def __init__(
        self,
        session: PreparedSession,
        indices: Optional[np.ndarray] = None,
        manual_branch: str = "raw",
    ) -> None:
        self.session = session
        self.indices = (
            np.arange(len(session.labels), dtype=np.int64)
            if indices is None
            else np.asarray(indices, dtype=np.int64)
        )
        self.branch_id = 0 if manual_branch == "raw" else 1

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        source_index = int(self.indices[index])
        return {
            "raw_eeg": torch.from_numpy(
                self.session.raw_eeg[source_index]
            ).unsqueeze(0),
            "cwt_image": torch.from_numpy(
                self.session.cwt_images[source_index]
            ),
            "graph_node_features": torch.from_numpy(
                self.session.graph_node_features[source_index]
            ),
            "graph_adjacency": torch.from_numpy(
                self.session.graph_adjacency[source_index]
            ),
            "cz_signal": torch.from_numpy(
                self.session.cz_signals[source_index]
            ),
            "label": torch.tensor(
                int(self.session.labels[source_index]),
                dtype=torch.long,
            ),
            "branch_selection": torch.tensor(
                self.branch_id,
                dtype=torch.long,
            ),
        }


def _annotation_code(description: str) -> Optional[int]:
    description = str(description).strip()
    if description.isdigit():
        return int(description)
    match = re.search(r"(\d+)\s*$", description)
    return int(match.group(1)) if match else None


def _events_by_code(raw: mne.io.BaseRaw) -> Dict[int, np.ndarray]:
    events, description_to_id = mne.events_from_annotations(
        raw,
        verbose=False,
    )
    output: Dict[int, np.ndarray] = {}
    for description, event_id in description_to_id.items():
        code = _annotation_code(description)
        if code is not None:
            output[code] = events[events[:, 2] == int(event_id)]
    return output


def _load_mat_labels(path: Path) -> np.ndarray:
    values = loadmat(str(path))
    if "classlabel" not in values:
        raise KeyError("{} does not contain classlabel.".format(path))
    labels = np.asarray(
        values["classlabel"],
        dtype=np.int64,
    ).reshape(-1)
    if not np.all(np.isin(labels, [1, 2, 3, 4])):
        raise ValueError("Unexpected labels in {}.".format(path))
    return labels - 1


def _labels_from_training_cues(
    events: Dict[int, np.ndarray],
) -> np.ndarray:
    ordered = []
    for code, label in TRAIN_CUE_TO_LABEL.items():
        if code not in events:
            raise ValueError(
                "Training GDF is missing event code {}.".format(code)
            )
        for event in events[code]:
            ordered.append((int(event[0]), label))
    ordered.sort(key=lambda item: item[0])
    return np.asarray(
        [label for _, label in ordered],
        dtype=np.int64,
    )


def validate_session_files(
    data_folder: Path,
    subject_id: int,
    session: str,
) -> Tuple[Path, Optional[Path]]:
    session = session.upper()
    if session not in {"T", "E"}:
        raise ValueError("session must be T or E.")

    gdf_path = data_folder / "A{:02d}{}.gdf".format(
        subject_id,
        session,
    )
    mat_path = gdf_path.with_suffix(".mat")

    if not gdf_path.is_file():
        raise FileNotFoundError("Missing recording: {}".format(gdf_path))
    if session == "E" and not mat_path.is_file():
        raise FileNotFoundError(
            "Evaluation labels are required: {}".format(mat_path)
        )
    return gdf_path, mat_path if mat_path.is_file() else None


def _replace_nonfinite(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data, dtype=np.float64).copy()
    for channel_index in range(data.shape[0]):
        channel = data[channel_index]
        invalid = ~np.isfinite(channel)
        if invalid.any():
            valid = channel[~invalid]
            if valid.size == 0:
                raise ValueError(
                    "Channel {} has no finite samples.".format(channel_index)
                )
            channel[invalid] = float(valid.mean())
    return data


def _bandpass(
    data: np.ndarray,
    sampling_rate: float,
    low_hz: float,
    high_hz: float,
    order: int,
) -> np.ndarray:
    nyquist = sampling_rate / 2.0
    sos = butter(
        order,
        [low_hz / nyquist, high_hz / nyquist],
        btype="bandpass",
        output="sos",
    )
    return sosfiltfilt(sos, data, axis=-1)


def _pick_eeg_channels(raw: mne.io.BaseRaw) -> Sequence[str]:
    eeg_names = [
        name
        for name in raw.ch_names
        if name not in EOG_CHANNELS
    ]
    if len(eeg_names) < 22:
        raise ValueError(
            "Expected at least 22 EEG channels, found {}.".format(
                len(eeg_names)
            )
        )
    return eeg_names[:22]


def load_raw_trials(
    config: ExperimentConfig,
    subject_id: int,
    session: str,
) -> Tuple[np.ndarray, np.ndarray, float, Tuple[str, ...]]:
    gdf_path, mat_path = validate_session_files(
        config.data_folder,
        subject_id,
        session,
    )
    raw = mne.io.read_raw_gdf(
        str(gdf_path),
        preload=True,
        verbose="WARNING",
    )
    raw._data[:] = _replace_nonfinite(raw.get_data())

    events = _events_by_code(raw)
    session = session.upper()

    if mat_path is not None:
        labels = _load_mat_labels(mat_path)
    elif session == "T":
        labels = _labels_from_training_cues(events)
    else:
        raise RuntimeError("Evaluation labels are unavailable.")

    if TRIAL_START_CODE in events:
        reference_samples = events[TRIAL_START_CODE][:, 0].astype(
            np.int64
        )
        relative_window_s = config.trial_window_s
    else:
        if session == "T":
            cue_events = np.concatenate(
                [events[code] for code in TRAIN_CUE_TO_LABEL],
                axis=0,
            )
            cue_events = cue_events[np.argsort(cue_events[:, 0])]
        else:
            if EVALUATION_CUE_CODE not in events:
                raise ValueError("No evaluation cue events were found.")
            cue_events = events[EVALUATION_CUE_CODE]
        reference_samples = cue_events[:, 0].astype(np.int64)
        relative_window_s = (
            config.trial_window_s[0] - 2.0,
            config.trial_window_s[1] - 2.0,
        )

    if len(reference_samples) != len(labels):
        raise ValueError(
            "Event/label mismatch: {} events vs {} labels.".format(
                len(reference_samples),
                len(labels),
            )
        )

    eeg_names = tuple(_pick_eeg_channels(raw))
    eeg_indices = [raw.ch_names.index(name) for name in eeg_names]
    data = raw.get_data(picks=eeg_indices)

    # Re-reference across all 22 EEG channels.
    data = data - data.mean(axis=0, keepdims=True)

    sampling_rate = float(raw.info["sfreq"])
    data = _bandpass(
        data,
        sampling_rate,
        config.preprocessing_band_hz[0],
        config.preprocessing_band_hz[1],
        config.filter_order,
    )

    start_offset = int(round(relative_window_s[0] * sampling_rate))
    stop_offset = int(round(relative_window_s[1] * sampling_rate))
    expected_length = stop_offset - start_offset

    trials = []
    retained_labels = []
    for sample, label in zip(reference_samples, labels):
        reference_index = int(sample) - int(raw.first_samp)
        start = reference_index + start_offset
        stop = reference_index + stop_offset

        if start < 0 or stop > data.shape[1]:
            LOGGER.warning(
                "Skipping out-of-range trial at sample %d.",
                sample,
            )
            continue

        trial = data[:, start:stop]
        if trial.shape[-1] != expected_length:
            raise RuntimeError(
                "Unexpected trial length {}.".format(trial.shape[-1])
            )
        trials.append(trial.astype(np.float32, copy=False))
        retained_labels.append(int(label))

    return (
        np.stack(trials, axis=0),
        np.asarray(retained_labels, dtype=np.int64),
        sampling_rate,
        eeg_names,
    )


def _cache_path(
    config: ExperimentConfig,
    subject_id: int,
    session: str,
) -> Path:
    return config.cache_folder / (
        "had_gcn_subject_{:02d}_{}_window_{}-{}_cwt_{}.npz".format(
            subject_id,
            session.upper(),
            config.trial_window_s[0],
            config.trial_window_s[1],
            config.cwt_image_size,
        )
    )


def prepare_session(
    config: ExperimentConfig,
    subject_id: int,
    session: str,
) -> PreparedSession:
    cache_path = _cache_path(config, subject_id, session)
    if cache_path.is_file() and not config.rebuild_cache:
        LOGGER.info("Loading cache: %s", cache_path)
        values = np.load(cache_path, allow_pickle=False)
        return PreparedSession(
            raw_eeg=values["raw_eeg"].astype(np.float32),
            cwt_images=values["cwt_images"].astype(np.float32),
            graph_node_features=values[
                "graph_node_features"
            ].astype(np.float32),
            graph_adjacency=values[
                "graph_adjacency"
            ].astype(np.float32),
            cz_signals=values["cz_signals"].astype(np.float32),
            labels=values["labels"].astype(np.int64),
            sampling_rate=float(values["sampling_rate"]),
            channel_names=tuple(values["channel_names"].tolist()),
        )

    raw_eeg, labels, sampling_rate, channel_names = load_raw_trials(
        config,
        subject_id,
        session,
    )

    channel_to_index = {
        channel: index
        for index, channel in enumerate(channel_names)
    }
    missing = [
        channel
        for channel in config.cwt_selected_channels
        if channel not in channel_to_index
    ]
    if missing:
        raise ValueError(
            "Missing CWT channels {}. Available channels: {}".format(
                missing,
                channel_names,
            )
        )

    cwt_indices = [
        channel_to_index[channel]
        for channel in config.cwt_selected_channels
    ]
    cwt_trials = raw_eeg[:, cwt_indices, :]
    cwt_images = build_tfi_dataset(
        cwt_trials,
        sampling_rate,
        config.cwt_frequency_hz,
        config.cwt_frequency_bins,
        config.cwt_morlet_cycles,
        config.cwt_image_size,
        config.cwt_workers,
    )

    # Supervised FBCSP is fitted later using only the training split.
    node_features = np.zeros(
        (
            raw_eeg.shape[0],
            raw_eeg.shape[1],
            len(config.graph_subbands_hz)
            * config.graph_features_per_band,
        ),
        dtype=np.float32,
    )
    adjacency = build_pli_adjacencies(
        raw_eeg,
        top_k=config.pli_top_k,
        min_weight=config.pli_min_weight,
    )

    cz_index = channel_to_index.get("EEG-Cz")
    if cz_index is None:
        raise ValueError("EEG-Cz is required for calibration.")
    cz_signals = raw_eeg[:, cz_index, :].astype(np.float32)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        raw_eeg=raw_eeg.astype(np.float32),
        cwt_images=cwt_images.astype(np.float16),
        graph_node_features=node_features.astype(np.float32),
        graph_adjacency=adjacency.astype(np.float16),
        cz_signals=cz_signals.astype(np.float32),
        labels=labels.astype(np.int64),
        sampling_rate=np.asarray(sampling_rate, dtype=np.float64),
        channel_names=np.asarray(channel_names),
    )
    LOGGER.info("Saved cache: %s", cache_path)

    return PreparedSession(
        raw_eeg=raw_eeg,
        cwt_images=cwt_images,
        graph_node_features=node_features,
        graph_adjacency=adjacency,
        cz_signals=cz_signals,
        labels=labels,
        sampling_rate=sampling_rate,
        channel_names=channel_names,
    )


def concatenate_sessions(
    sessions: Sequence[PreparedSession],
) -> PreparedSession:
    if not sessions:
        raise ValueError("At least one session is required.")

    reference_channels = sessions[0].channel_names
    reference_rate = sessions[0].sampling_rate
    for session in sessions[1:]:
        if session.channel_names != reference_channels:
            raise ValueError("Channel orders differ across sessions.")
        if abs(session.sampling_rate - reference_rate) > 1e-6:
            raise ValueError("Sampling rates differ across sessions.")

    return PreparedSession(
        raw_eeg=np.concatenate([item.raw_eeg for item in sessions]),
        cwt_images=np.concatenate([item.cwt_images for item in sessions]),
        graph_node_features=np.concatenate(
            [item.graph_node_features for item in sessions]
        ),
        graph_adjacency=np.concatenate(
            [item.graph_adjacency for item in sessions]
        ),
        cz_signals=np.concatenate([item.cz_signals for item in sessions]),
        labels=np.concatenate([item.labels for item in sessions]),
        sampling_rate=reference_rate,
        channel_names=reference_channels,
    )


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    device: torch.device,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
