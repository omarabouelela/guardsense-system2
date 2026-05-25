"""Unified indexing helpers for Trigger and Verifier sources."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.data.pose_preprocess import DatasetIndexRecord, index_pose_sources
from src.data.video_preprocess import VideoIndexRecord, index_video_sources


def index_trigger_sources(config_sources: list[dict[str, Any]]) -> list[DatasetIndexRecord]:
    """Index configured trigger pose sources."""
    rows: list[DatasetIndexRecord] = []
    for source in config_sources:
        rows.extend(
            index_pose_sources(
                source_dirs=[Path(p) for p in source["paths"]],
                source_dataset=str(source["name"]),
                label=int(source["label"]),
                synthetic_or_real=str(source.get("synthetic_or_real", "unknown")),
                split=str(source.get("split", "unspecified")),
            )
        )
    return rows


def index_verifier_sources(config_sources: list[dict[str, Any]]) -> list[VideoIndexRecord]:
    """Index configured verifier video sources."""
    rows: list[VideoIndexRecord] = []
    for source in config_sources:
        rows.extend(
            index_video_sources(
                source_dirs=[Path(p) for p in source["paths"]],
                source_dataset=str(source["name"]),
                label=source.get("label"),
                synthetic_or_real=str(source.get("synthetic_or_real", "unknown")),
                split=str(source.get("split", "unspecified")),
            )
        )
    return rows
