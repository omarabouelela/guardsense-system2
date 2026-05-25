"""Reusable split manager with reproducible stratification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.data.trigger_dataset import SplitConfig, stratified_split_indices
from src.data.video_preprocess import stratified_split


@dataclass(slots=True)
class SplitResult:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


def split_trigger(labels: Sequence[int], groups: Sequence[str] | None, config: SplitConfig) -> SplitResult:
    """Split pose labels into train/val/test indices."""
    tr, va, te = stratified_split_indices(labels=labels, groups=groups, config=config)
    return SplitResult(tr, va, te)


def split_verifier(
    labels: Sequence[int],
    groups: Sequence[str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    source_aware: bool,
) -> SplitResult:
    """Split verifier clips using existing video split utility."""
    tr, va, te = stratified_split(
        labels=labels,
        groups=groups,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        source_aware=source_aware,
    )
    return SplitResult(tr, va, te)
