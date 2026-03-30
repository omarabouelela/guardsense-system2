from __future__ import annotations

from pathlib import Path

import numpy as np

from src.common.runtime_utils import latest_checkpoint, resolve_checkpoint_path, resolve_raw_data_path
from src.data.trigger_dataset import SplitConfig, stratified_split_indices
from src.data.video_preprocess import stratified_split


def test_trigger_split_empty_indices_are_int64() -> None:
    labels = [0, 0, 0]
    cfg = SplitConfig(train_ratio=1.0, val_ratio=0.0, test_ratio=0.0, seed=1)
    train_idx, val_idx, test_idx = stratified_split_indices(labels, groups=None, config=cfg)
    assert train_idx.dtype == np.int64
    assert val_idx.dtype == np.int64
    assert test_idx.dtype == np.int64


def test_verifier_split_empty_indices_are_int64() -> None:
    labels = [0, 0, 0]
    groups = ["a", "a", "a"]
    train_idx, val_idx, test_idx = stratified_split(
        labels=labels,
        groups=groups,
        train_ratio=1.0,
        val_ratio=0.0,
        test_ratio=0.0,
        seed=1,
        source_aware=False,
    )
    assert train_idx.dtype == np.int64
    assert val_idx.dtype == np.int64
    assert test_idx.dtype == np.int64


def test_resolve_raw_data_path_compat(tmp_path: Path) -> None:
    dataset_raw = tmp_path / "dataset" / "raw" / "rwf2000"
    dataset_raw.mkdir(parents=True)
    original_cwd = Path.cwd()
    try:
        # simulate project-relative lookup
        import os

        os.chdir(tmp_path)
        resolved = resolve_raw_data_path("data/raw/rwf2000")
        assert resolved == Path("dataset/raw/rwf2000")
    finally:
        os.chdir(original_cwd)


def test_resolve_checkpoint_path_latest(tmp_path: Path) -> None:
    base = tmp_path / "artifacts" / "runs"
    run = base / "trigger_temporal_cnn_20260101T000000Z"
    run.mkdir(parents=True)
    ckpt = run / "best_model.pt"
    ckpt.write_text("ok", encoding="utf-8")

    resolved = resolve_checkpoint_path(base / "latest" / "best_model.pt", base_dir=base, run_prefix="trigger_")
    assert resolved == ckpt
    assert latest_checkpoint(base_dir=base, run_prefix="trigger_") == ckpt
