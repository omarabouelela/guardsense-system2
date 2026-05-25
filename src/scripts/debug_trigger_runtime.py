"""Debug Trigger runtime preprocessing and event aggregation for one pose file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from ._bootstrap import ensure_project_root_on_path
except ImportError:  # direct script execution
    from _bootstrap import ensure_project_root_on_path

ensure_project_root_on_path()

from src.data.pose_preprocess import NormalizationConfig, SequenceAssemblyConfig
from src.trigger.infer import InferenceConfig, TriggerInferencer


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Debug Trigger runtime inference for one pose file")
    parser.add_argument("--model-path", type=str, required=True, help="Trigger checkpoint path")
    parser.add_argument("--pose-file", type=Path, required=True, help="Pose file path (.txt, .npy, .hdf5)")
    parser.add_argument("--device", type=str, default="auto", help="Device string: auto, cpu, cuda")
    parser.add_argument("--normalization-mode", choices=["none", "frame", "bbox"], default="frame")
    parser.add_argument("--frame-width", type=int, default=1920)
    parser.add_argument("--frame-height", type=int, default=1080)
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--short-policy", choices=["pad", "drop", "truncate"], default="pad")
    parser.add_argument("--fight-class-id", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--include-window-debug", action="store_true")
    return parser


def main() -> None:
    """CLI entrypoint."""
    args = build_parser().parse_args()
    inferencer = TriggerInferencer(
        InferenceConfig(
            model_path=args.model_path,
            device=args.device,
            normalization=NormalizationConfig(
                mode=args.normalization_mode,
                frame_width=args.frame_width,
                frame_height=args.frame_height,
            ),
            temporal=SequenceAssemblyConfig(
                window_size=args.window_size,
                overlap=args.overlap,
                short_policy=args.short_policy,
            ),
            fight_class_id=args.fight_class_id,
            top_k=args.top_k,
            include_window_debug=args.include_window_debug,
        )
    )
    result = inferencer.infer_pose_file(args.pose_file)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
