"""CLI runtime for GuardSense dual-model fusion inference."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import yaml

try:
    from ._bootstrap import ensure_project_root_on_path
except ImportError:  # direct script execution
    from _bootstrap import ensure_project_root_on_path

ensure_project_root_on_path()

from src.common.runtime_utils import resolve_checkpoint_path
from src.data.video_preprocess import VideoProcessConfig
from src.fusion.decision_logic import FusionRuntime, FusionRuntimeConfig, FusionThresholds, RuntimeStats, update_stats
from src.fusion.event_schema import FusionDecision, FusionEvent, load_json_events, load_manifest_csv
from src.fusion.frigate_adapter import FrigateAdapter, FrigateAdapterConfig
from src.trigger.infer import InferenceConfig, TriggerInferencer
from src.verifier.infer import VerifierInferenceConfig, VerifierInferencer

LOGGER = logging.getLogger(__name__)


class NullInferencer:
    """Dry-run-safe inferencer stub."""

    def infer_frigate_event(self, *_: Any, **__: Any) -> dict[str, Any]:
        raise RuntimeError("NullInferencer should not run outside dry-run mode")

    def infer_clip(self, *_: Any, **__: Any) -> dict[str, Any]:
        raise RuntimeError("NullInferencer should not run outside dry-run mode")


def build_parser() -> argparse.ArgumentParser:
    """Build parser for online/offline dual inference runtime."""
    parser = argparse.ArgumentParser(description="Run GuardSense Trigger+Verifier fusion inference")
    parser.add_argument("--config", type=Path, required=True, help="Fusion runtime YAML config")

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--event-id", type=str, help="Frigate event ID (resolved from API)")
    source.add_argument("--event-json", type=Path, help="Path to one JSON event payload or {events: [...]} file")
    source.add_argument("--events-dir", type=Path, help="Folder containing *.json event payload files")
    source.add_argument("--manifest-csv", type=Path, help="Offline event manifest CSV")

    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for logs and artifacts")
    parser.add_argument("--dry-run", action="store_true", help="Parse/resolve events without model execution")
    parser.add_argument("--debug", action="store_true", help="Enable verbose DEBUG logging")
    return parser


def setup_logging(output_dir: Path, debug: bool) -> None:
    """Configure console and file logging."""
    output_dir.mkdir(parents=True, exist_ok=True)
    run_log = output_dir / "run.log"
    level = logging.DEBUG if debug else logging.INFO

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")

    stream = logging.StreamHandler()
    stream.setLevel(level)
    stream.setFormatter(formatter)
    root.addHandler(stream)

    file_handler = logging.FileHandler(run_log, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)


def load_runtime_config(config_path: Path, output_dir: Path) -> tuple[FusionRuntimeConfig, FrigateAdapterConfig, dict[str, Any]]:
    """Load YAML and map to runtime dataclasses."""
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    thresholds = FusionThresholds(**raw.get("thresholds", {}))
    vp = raw.get("video_process", {})
    process_cfg = VideoProcessConfig(
        target_fps=vp.get("target_fps", 30),
        target_duration_seconds=vp.get("target_duration_seconds", 3.0),
        min_duration_seconds=vp.get("min_duration_seconds", 2.0),
        max_duration_seconds=vp.get("max_duration_seconds", 4.0),
        target_width=vp.get("target_width", 640),
        target_height=vp.get("target_height", 360),
        target_codec=vp.get("target_codec", "libx264"),
        crf=vp.get("crf", 23),
        preset=vp.get("preset", "fast"),
    )

    extraction_dir = output_dir / raw.get("runtime", {}).get("extraction_dir", "extracted_clips")
    runtime = FusionRuntimeConfig(
        extraction_output_dir=extraction_dir,
        video_process=process_cfg,
        thresholds=thresholds,
        drop_trigger_normal_early=raw.get("runtime", {}).get("drop_trigger_normal_early", True),
    )
    adapter_cfg = FrigateAdapterConfig(
        api_base_url=raw.get("frigate", {}).get("api_base_url"),
        api_key=raw.get("frigate", {}).get("api_key"),
        clips_root=Path(raw.get("frigate", {}).get("clips_root", "/media/frigate/clips")),
        recordings_root=Path(raw.get("frigate", {}).get("recordings_root", "/media/frigate/recordings")),
    )
    return runtime, adapter_cfg, raw


def build_events(args: argparse.Namespace, adapter: FrigateAdapter) -> list[FusionEvent]:
    """Build event list from selected input source."""
    if args.event_id:
        return [adapter.resolve_event(args.event_id)]
    if args.event_json:
        return [adapter.resolve_media_paths(event) for event in load_json_events(args.event_json, source="manual")]
    if args.events_dir:
        events: list[FusionEvent] = []
        for path in sorted(args.events_dir.glob("*.json")):
            events.extend(load_json_events(path, source="filesystem"))
        return [adapter.resolve_media_paths(event) for event in events]
    if args.manifest_csv:
        return [adapter.resolve_media_paths(event) for event in load_manifest_csv(args.manifest_csv)]
    raise ValueError("No input source provided")


def build_inferencers(raw_cfg: dict[str, Any], dry_run: bool) -> tuple[Any, Any]:
    """Instantiate model wrappers while preserving existing interfaces."""
    if dry_run:
        return NullInferencer(), NullInferencer()

    trigger_model_path = raw_cfg.get("models", {}).get("trigger_model_path")
    verifier_model_path = raw_cfg.get("models", {}).get("verifier_model_path")
    if not trigger_model_path or not verifier_model_path:
        raise ValueError("models.trigger_model_path and models.verifier_model_path are required unless --dry-run")

    trigger_model_path = str(
        resolve_checkpoint_path(trigger_model_path, base_dir="artifacts/runs", run_prefix="trigger_")
    )
    verifier_model_path = str(
        resolve_checkpoint_path(verifier_model_path, base_dir="artifacts/verifier_runs", run_prefix="verifier_")
    )

    trigger = TriggerInferencer(
        InferenceConfig(
            model_path=trigger_model_path,
            device=raw_cfg.get("runtime", {}).get("device", "auto"),
        )
    )
    verifier = VerifierInferencer(
        VerifierInferenceConfig(
            model_path=verifier_model_path,
            num_frames=raw_cfg.get("verifier_inference", {}).get("num_frames", 16),
            temporal_stride=raw_cfg.get("verifier_inference", {}).get("temporal_stride", 2),
            device=raw_cfg.get("runtime", {}).get("device", "auto"),
        )
    )
    return trigger, verifier


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    """Append one JSON line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_summary(path: Path, stats: RuntimeStats) -> None:
    """Write run summary JSON."""
    payload = {
        "number_of_events_processed": stats.processed,
        "number_dropped_at_trigger": stats.dropped_at_trigger,
        "number_escalated_to_verifier": stats.escalated_to_verifier,
        "number_finalized_as_0": stats.finalized_0,
        "number_finalized_as_1": stats.finalized_1,
        "number_finalized_as_2": stats.finalized_2,
        "number_with_missing_media": stats.missing_media,
        "number_requiring_manual_review": stats.review_needed,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    """Run runtime pipeline and persist standard artifacts."""
    setup_logging(args.output_dir, debug=args.debug)
    runtime_cfg, adapter_cfg, raw_cfg = load_runtime_config(args.config, args.output_dir)
    adapter = FrigateAdapter(adapter_cfg)
    events = build_events(args, adapter)

    trigger, verifier = build_inferencers(raw_cfg, dry_run=args.dry_run)
    runtime = FusionRuntime(trigger_inferencer=trigger, verifier_inferencer=verifier, config=runtime_cfg)

    processed_path = args.output_dir / "processed_events.jsonl"
    failed_path = args.output_dir / "failed_events.jsonl"
    review_path = args.output_dir / "review_needed.jsonl"
    summary_path = args.output_dir / "summary.json"

    stats = RuntimeStats()
    keyed_output: dict[str, dict[str, Any]] = {}

    for event in events:
        try:
            decision: FusionDecision = runtime.process_event(event, dry_run=args.dry_run)
            keyed_output.update(decision.keyed_json())
            append_jsonl(processed_path, decision.to_dict())
            update_stats(stats, decision)
            if decision.review_needed:
                append_jsonl(review_path, decision.to_dict())
            if decision.status == "failed":
                append_jsonl(failed_path, decision.to_dict())
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed event %s", event.event_id)
            failure = {
                "event_id": event.event_id,
                "camera_name": event.camera_name,
                "status": "failed",
                "error": str(exc),
                "source": event.source,
            }
            append_jsonl(failed_path, failure)
            stats.processed += 1
            stats.missing_media += 1

    write_summary(summary_path, stats)
    (args.output_dir / "final_output_keyed.json").write_text(json.dumps(keyed_output, indent=2), encoding="utf-8")

    LOGGER.info("Processed %d event(s)", stats.processed)
    LOGGER.info("Summary written to %s", summary_path)
    return 0


def main() -> None:
    """CLI entrypoint."""
    args = build_parser().parse_args()
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
