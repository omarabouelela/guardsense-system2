"""Dataset utilities for GuardSense Verifier RGB model."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class VerifierDatasetConfig:
    """Sampling settings for clip loading."""

    num_frames: int = 16
    temporal_stride: int = 2
    random_sample: bool = True


class VerifierVideoDataset(Dataset[tuple[torch.Tensor, torch.Tensor, str]]):
    """Torch dataset reading RGB clips from metadata manifest."""

    def __init__(
        self,
        manifest_path: Path,
        split: str,
        config: VerifierDatasetConfig,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        self.df = pd.read_csv(manifest_path)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        if self.df.empty:
            LOGGER.warning("No samples found for split=%s in %s; continuing with empty dataset.", split, manifest_path)
        self.config = config
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        row = self.df.iloc[idx]
        clip_path = Path(row["processed_path"])
        label = int(row["label"])
        video = load_video_tensor(clip_path)
        sampled = sample_clip_frames(
            video,
            num_frames=self.config.num_frames,
            temporal_stride=self.config.temporal_stride,
            random_sample=self.config.random_sample,
        )
        if self.transform is not None:
            sampled = self.transform(sampled)
        return sampled, torch.tensor(label, dtype=torch.long), str(row.get("clip_id", clip_path.stem))


def load_video_tensor(clip_path, num_frames=16, resize_hw=(112, 112)):
    import cv2
    import torch
    import numpy as np
    from pathlib import Path

    path = Path(clip_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        frame_indices = list(range(num_frames))
    else:
        frame_indices = np.linspace(0, max(total_frames - 1, 0), num_frames).astype(int).tolist()

    frames = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()

        if not ok or frame is None:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, resize_hw)
        frames.append(frame)

    cap.release()

    if not frames:
        raise RuntimeError(f"No frames decoded from video: {path}")

    while len(frames) < num_frames:
        frames.append(frames[-1])

    video = np.stack(frames[:num_frames], axis=0)  # T, H, W, C
    video = torch.from_numpy(video).float() / 255.0

    # Convert to C, T, H, W for video models
    video = video.permute(3, 0, 1, 2).contiguous()

    return video


def sample_clip_frames(
    video: torch.Tensor,
    num_frames: int,
    temporal_stride: int,
    random_sample: bool,
) -> torch.Tensor:
    """Sample fixed-length temporal clip from [C,T,H,W]."""
    if video.ndim != 4:
        raise ValueError(f"Expected [C,T,H,W], got {video.shape}")
    _, t, _, _ = video.shape
    effective = num_frames * temporal_stride

    if t >= effective:
        max_start = t - effective
        if random_sample:
            start = int(np.random.randint(0, max_start + 1))
        else:
            start = max_start // 2
        idx = torch.arange(start, start + effective, temporal_stride)
        return video[:, idx, :, :]

    # pad by repeating last frame.
    idx = torch.arange(0, t, temporal_stride)
    sampled = video[:, idx, :, :] if idx.numel() > 0 else video[:, :1, :, :]
    while sampled.shape[1] < num_frames:
        sampled = torch.cat([sampled, sampled[:, -1:, :, :]], dim=1)
    return sampled[:, :num_frames, :, :]


def build_train_transform() -> Callable[[torch.Tensor], torch.Tensor]:
    """Simple surveillance-appropriate augmentations for train split."""

    def _transform(x: torch.Tensor) -> torch.Tensor:
        # mild brightness jitter + horizontal flip for robustness.
        if torch.rand(1).item() < 0.5:
            x = torch.flip(x, dims=[3])
        if torch.rand(1).item() < 0.5:
            alpha = float(torch.empty(1).uniform_(0.85, 1.15).item())
            x = torch.clamp(x * alpha, 0.0, 1.0)
        return x

    return _transform


def build_eval_transform() -> Callable[[torch.Tensor], torch.Tensor]:
    """Identity transform for validation/test."""

    def _transform(x: torch.Tensor) -> torch.Tensor:
        return x

    return _transform


def class_weights_from_manifest(manifest_path: Path, split: str, num_classes: int | None = None) -> torch.Tensor:
    """Compute inverse-frequency class weights for CrossEntropyLoss."""
    df = pd.read_csv(manifest_path)
    counts = df[df["split"] == split]["label"].value_counts().sort_index()
    class_count = int(num_classes if num_classes is not None else (int(df["label"].max()) + 1))
    weights = []
    total = float(counts.sum())
    for class_id in range(class_count):
        c = float(counts.get(class_id, 1.0))
        weights.append(total / (float(class_count) * c))
    return torch.tensor(weights, dtype=torch.float32)
