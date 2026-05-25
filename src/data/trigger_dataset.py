"""Dataset abstractions and split helpers for Trigger model."""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class SplitConfig:
    """Config for deterministic split generation."""

    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42
    source_aware: bool = False


class TriggerPoseDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Torch dataset for (T,K,C) pose sequences and integer labels."""

    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        if features.ndim != 4:
            raise ValueError(f"features must be 4D [N,T,K,C], got {features.shape}")
        if labels.ndim != 1:
            raise ValueError(f"labels must be 1D [N], got {labels.shape}")
        if len(features) != len(labels):
            raise ValueError("features and labels length mismatch")
        self.features = torch.from_numpy(features).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self) -> int:
        return self.features.size(0)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


def stratified_split_indices(
    labels: Sequence[int],
    groups: Sequence[str] | None,
    config: SplitConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic stratified split with optional source-aware grouping."""
    labels_arr = np.asarray(labels, dtype=np.int64)
    rng = np.random.default_rng(config.seed)

    unique_labels = np.unique(labels_arr)
    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []

    for class_id in unique_labels:
        class_indices = np.where(labels_arr == class_id)[0]
        if class_indices.size == 0:
            continue

        if config.source_aware and groups is not None:
            class_groups = np.asarray(groups)[class_indices]
            unique_groups = np.unique(class_groups)
            rng.shuffle(unique_groups)
            sorted_indices: list[int] = []
            for group in unique_groups:
                sorted_indices.extend(class_indices[class_groups == group].tolist())
            class_indices = np.asarray(sorted_indices, dtype=np.int64)
        else:
            rng.shuffle(class_indices)

        n_total = class_indices.size
        n_train = max(1, int(round(n_total * config.train_ratio)))
        n_val = max(1, int(round(n_total * config.val_ratio)))
        n_train = min(n_train, n_total)
        n_val = min(n_val, max(0, n_total - n_train))
        n_test = max(0, n_total - n_train - n_val)

        train_idx.extend(class_indices[:n_train])
        val_idx.extend(class_indices[n_train : n_train + n_val])
        if n_test > 0:
            test_idx.extend(class_indices[-n_test:])

    for split in (train_idx, val_idx, test_idx):
        rng.shuffle(split)

    LOGGER.info("Split counts train=%d val=%d test=%d", len(train_idx), len(val_idx), len(test_idx))
    return np.asarray(train_idx, dtype=np.int64), np.asarray(val_idx, dtype=np.int64), np.asarray(test_idx, dtype=np.int64)


def save_split_manifest(
    output_path: Path,
    sample_ids: Sequence[str],
    labels: Sequence[int],
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    test_indices: np.ndarray,
) -> None:
    """Save split assignments as CSV manifest."""
    split_by_idx: dict[int, str] = {}
    for idx in train_indices.tolist():
        split_by_idx[idx] = "train"
    for idx in val_indices.tolist():
        split_by_idx[idx] = "val"
    for idx in test_indices.tolist():
        split_by_idx[idx] = "test"

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "label", "split"])
        writer.writeheader()
        for idx, sample_id in enumerate(sample_ids):
            writer.writerow({"sample_id": sample_id, "label": int(labels[idx]), "split": split_by_idx.get(idx, "unused")})


def subset_arrays(features: np.ndarray, labels: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Small helper for split selection."""
    return features[indices], labels[indices]
