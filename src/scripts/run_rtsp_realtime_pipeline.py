"""Prototype rolling real-time RTSP runner for GuardSense Trigger + Verifier.

This script is intentionally separate from the Frigate event runner. It is a
development-safe live loop: no sirens, webhooks, or external alerts are fired.
It reads an RTSP stream, keeps a short rolling RGB frame buffer, runs Trigger on
periodic pose extraction, and runs Verifier only for high Trigger scores.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlsplit, urlunsplit

import yaml

try:
    from ._bootstrap import ensure_project_root_on_path
except ImportError:  # direct script execution
    from _bootstrap import ensure_project_root_on_path

ensure_project_root_on_path()

from src.data.pose_extraction import PoseExtractorConfig, extract_video_pose, load_yolo_model

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class RealtimeThresholds:
    """Automatic RTSP decision thresholds."""

    trigger_medium: float = 0.55
    trigger_high: float = 0.75
    verifier_alert: float = 0.82


@dataclass(slots=True)
class RealtimeLoopConfig:
    """Rolling loop controls for the prototype runner."""

    buffer_seconds: float = 5.0
    min_window_seconds: float = 3.0
    analysis_interval_seconds: float = 2.0
    reconnect_delay_seconds: float = 3.0
    output_fps: float = 30.0
    thresholds: RealtimeThresholds = field(default_factory=RealtimeThresholds)


@dataclass(slots=True)
class RuntimeModels:
    """Model objects used by the live loop."""

    trigger: Any
    verifier: Any
    pose_model: Any
    pose_config: PoseExtractorConfig


@dataclass(slots=True)
class LiveStats:
    """Counters persisted to summary.json on shutdown."""

    started_at: str
    stopped_at: str | None = None
    stop_reason: str = "running"
    frames_read: int = 0
    reconnect_attempts: int = 0
    windows_processed: int = 0
    normal_background: int = 0
    low_confidence_suspicious_motion: int = 0
    trigger_positive: int = 0
    confirmed_suspicious: int = 0
    verifier_rejected: int = 0
    pose_extraction_failed: int = 0
    trigger_failed: int = 0
    verifier_failed: int = 0


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Run prototype rolling RTSP GuardSense inference")
    parser.add_argument("--config", type=Path, required=True, help="Fusion runtime YAML config")
    parser.add_argument("--rtsp-url", type=str, required=True, help="RTSP stream URL")
    parser.add_argument("--camera-name", type=str, required=True, help="Camera name for live event records")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output folder for JSONL, pose files, and alert clips")
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG logging")
    parser.add_argument("--max-seconds", type=float, default=None, help="Optional maximum runtime for local tests")
    parser.add_argument(
        "--trigger-high-threshold",
        type=float,
        default=0.75,
        help="Trigger class-1 probability required to save a clip and run Verifier",
    )
    parser.add_argument(
        "--verifier-alert-threshold",
        type=float,
        default=0.82,
        help="Verifier class-1 probability required to write an automatic alert",
    )
    return parser


def mask_rtsp_url(rtsp_url: str) -> str:
    """Mask credentials in an RTSP URL before logging or printing."""
    split = urlsplit(rtsp_url)
    if "@" not in split.netloc:
        return rtsp_url
    _, _, host_part = split.netloc.rpartition("@")
    return urlunsplit((split.scheme, f"***:***@{host_part}", split.path, split.query, split.fragment))


def safe_filename_token(value: str) -> str:
    """Return a conservative filename token."""
    cleaned = "".join(ch if ch.isascii() and (ch.isalnum() or ch in {"-", "_", "."}) else "_" for ch in value)
    return cleaned.strip("._-") or "camera"


def classify_realtime_decision(
    trigger_p1: float,
    verifier_p1: float | None,
    thresholds: RealtimeThresholds | None = None,
) -> str:
    """Classify one automatic real-time decision from Trigger/Verifier probabilities."""
    th = thresholds or RealtimeThresholds()
    if trigger_p1 < th.trigger_medium:
        return "normal_background"
    if trigger_p1 < th.trigger_high:
        return "low_confidence_suspicious_motion"
    if verifier_p1 is None:
        return "trigger_positive"
    if verifier_p1 >= th.verifier_alert:
        return "confirmed_suspicious"
    return "verifier_rejected"


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    """Append one JSONL record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def setup_logging(output_dir: Path, debug: bool) -> None:
    """Configure console and file logging for the live runner."""
    output_dir.mkdir(parents=True, exist_ok=True)
    level = logging.DEBUG if debug else logging.INFO
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    stream = logging.StreamHandler()
    stream.setLevel(level)
    stream.setFormatter(formatter)
    root.addHandler(stream)

    file_handler = logging.FileHandler(output_dir / "run.log", encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)


def load_yaml_config(path: Path) -> dict[str, Any]:
    """Load YAML config."""
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def build_loop_config(
    raw_cfg: dict[str, Any],
    trigger_high_threshold: float = 0.75,
    verifier_alert_threshold: float = 0.82,
) -> RealtimeLoopConfig:
    """Build RTSP loop config from optional YAML overrides."""
    rtsp_cfg = raw_cfg.get("rtsp_realtime", {}) or {}
    threshold_cfg = rtsp_cfg.get("thresholds", {}) or {}
    vp = raw_cfg.get("video_process", {}) or {}
    thresholds = RealtimeThresholds(
        trigger_medium=float(threshold_cfg.get("trigger_medium", 0.55)),
        trigger_high=float(trigger_high_threshold),
        verifier_alert=float(verifier_alert_threshold),
    )
    buffer_seconds = float(rtsp_cfg.get("buffer_seconds", 5.0))
    return RealtimeLoopConfig(
        buffer_seconds=buffer_seconds,
        min_window_seconds=float(rtsp_cfg.get("min_window_seconds", min(3.0, buffer_seconds))),
        analysis_interval_seconds=float(rtsp_cfg.get("analysis_interval_seconds", 2.0)),
        reconnect_delay_seconds=float(rtsp_cfg.get("reconnect_delay_seconds", 3.0)),
        output_fps=float(rtsp_cfg.get("output_fps", vp.get("target_fps", 30))),
        thresholds=thresholds,
    )


def build_pose_extractor_config(raw_cfg: dict[str, Any]) -> PoseExtractorConfig:
    """Build pose extraction config from optional fusion YAML settings."""
    raw_pose_cfg = raw_cfg.get("pose_extraction", {})
    pose_cfg = dict(raw_pose_cfg.get("extractor", raw_pose_cfg)) if isinstance(raw_pose_cfg, dict) else {}
    for ignored_key in ("enabled", "output_dir", "manifest_path", "log_file", "log_level", "datasets"):
        pose_cfg.pop(ignored_key, None)
    runtime_device = raw_cfg.get("runtime", {}).get("device")
    if runtime_device and "device" not in pose_cfg:
        pose_cfg["device"] = runtime_device
    return PoseExtractorConfig(**pose_cfg)


def build_runtime_models(raw_cfg: dict[str, Any]) -> RuntimeModels:
    """Instantiate Trigger, Verifier, and pose models."""
    from src.data.pose_preprocess import NormalizationConfig, SequenceAssemblyConfig
    from src.trigger.infer import InferenceConfig, TriggerInferencer
    from src.verifier.infer import VerifierInferenceConfig, VerifierInferencer

    trigger_model_path = raw_cfg.get("models", {}).get("trigger_model_path")
    verifier_model_path = raw_cfg.get("models", {}).get("verifier_model_path")
    if not trigger_model_path or not verifier_model_path:
        raise ValueError("models.trigger_model_path and models.verifier_model_path are required")

    trigger_inference_cfg = raw_cfg.get("trigger_inference", {}) or {}
    runtime_device = raw_cfg.get("runtime", {}).get("device", "auto")
    trigger = TriggerInferencer(
        InferenceConfig(
            model_path=str(trigger_model_path),
            device=runtime_device,
            normalization=NormalizationConfig(**trigger_inference_cfg.get("normalization", {})),
            temporal=SequenceAssemblyConfig(**trigger_inference_cfg.get("temporal", {})),
            fight_class_id=trigger_inference_cfg.get("fight_class_id", 1),
            top_k=trigger_inference_cfg.get("top_k", 3),
            include_window_debug=trigger_inference_cfg.get("include_window_debug", False),
        )
    )
    verifier = VerifierInferencer(
        VerifierInferenceConfig(
            model_path=str(verifier_model_path),
            num_frames=raw_cfg.get("verifier_inference", {}).get("num_frames", 16),
            temporal_stride=raw_cfg.get("verifier_inference", {}).get("temporal_stride", 2),
            device=runtime_device,
        )
    )
    pose_config = build_pose_extractor_config(raw_cfg)
    pose_model = load_yolo_model(pose_config.model)
    return RuntimeModels(trigger=trigger, verifier=verifier, pose_model=pose_model, pose_config=pose_config)


def open_capture(rtsp_url: str) -> Any | None:
    """Open an RTSP stream using OpenCV."""
    import cv2

    capture = cv2.VideoCapture(rtsp_url)
    if not capture.isOpened():
        capture.release()
        return None
    return capture


def write_rgb_clip(frames_rgb: Sequence[Any], output_path: Path, fps: float) -> None:
    """Write RGB frames to a short mp4 clip for pose/Verifier processing."""
    import cv2

    if not frames_rgb:
        raise ValueError("Cannot write an empty frame buffer")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames_rgb[0].shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(float(fps), 1.0),
        (int(width), int(height)),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open clip writer for {output_path}")
    try:
        for frame_rgb in frames_rgb:
            writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def frame_window_duration(buffer: deque[tuple[float, Any]]) -> float:
    """Return approximate seconds represented by the rolling buffer."""
    if len(buffer) < 2:
        return 0.0
    return max(buffer[-1][0] - buffer[0][0], 0.0)


def event_id_for(camera_name: str, sequence_id: int) -> str:
    """Create a timestamped event id for the live runner."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{safe_filename_token(camera_name)}_{stamp}_{sequence_id:06d}"


def trigger_p1_from_output(trigger_out: dict[str, Any]) -> float:
    """Extract binary fight/suspicious probability from Trigger output."""
    return float((trigger_out.get("class_probabilities") or {}).get("1", 0.0))


def verifier_p1_from_output(verifier_out: dict[str, Any]) -> float:
    """Extract binary fight/suspicious probability from Verifier output."""
    return float((verifier_out.get("class_probabilities") or {}).get("1", 0.0))


def maybe_remove(path: Path) -> None:
    """Best-effort cleanup for temporary clips."""
    try:
        if path.exists():
            path.unlink()
    except OSError:
        LOGGER.debug("Could not remove temporary file %s", path)


def process_window(
    *,
    event_id: str,
    camera_name: str,
    frames_rgb: Sequence[Any],
    output_dir: Path,
    loop_cfg: RealtimeLoopConfig,
    models: RuntimeModels,
    live_events_path: Path,
    alerts_path: Path,
    stats: LiveStats,
) -> None:
    """Run pose, Trigger, and optional Verifier on one rolling window."""
    tmp_clip_path = output_dir / "_tmp" / f"{event_id}_window.mp4"
    pose_path = output_dir / "extracted_pose" / f"{event_id}.txt"
    saved_clip_path = output_dir / "clips" / f"{event_id}.mp4"
    created_at = datetime.now(UTC).isoformat()
    stats.windows_processed += 1

    base_record: dict[str, Any] = {
        "event_id": event_id,
        "camera_name": camera_name,
        "created_at": created_at,
        "frame_count": len(frames_rgb),
        "pose_path": str(pose_path),
        "clip_path": None,
        "trigger_p1": None,
        "verifier_p1": None,
        "status": None,
        "notes": [],
    }

    try:
        write_rgb_clip(frames_rgb, tmp_clip_path, loop_cfg.output_fps)
        success, frame_count, detected_count, avg_people, note = extract_video_pose(
            model=models.pose_model,
            video_path=tmp_clip_path,
            output_path=pose_path,
            config=models.pose_config,
        )
    except Exception as exc:  # noqa: BLE001
        stats.pose_extraction_failed += 1
        record = {
            **base_record,
            "status": "pose_extraction_failed",
            "error": str(exc),
            "notes": ["pose extraction failed; trigger skipped"],
        }
        append_jsonl(live_events_path, record)
        print(f"{event_id} status=pose_extraction_failed")
        maybe_remove(tmp_clip_path)
        return

    if not success:
        stats.pose_extraction_failed += 1
        record = {
            **base_record,
            "status": "pose_extraction_failed",
            "notes": ["pose extraction failed; trigger skipped"],
            "pose_frame_count": frame_count,
            "pose_detected_frame_count": detected_count,
            "pose_avg_detected_persons": avg_people,
            "pose_note": note,
        }
        append_jsonl(live_events_path, record)
        print(f"{event_id} status=pose_extraction_failed note={note}")
        maybe_remove(tmp_clip_path)
        return

    try:
        trigger_out = models.trigger.infer_pose_file(pose_path)
        trigger_p1 = trigger_p1_from_output(trigger_out)
    except Exception as exc:  # noqa: BLE001
        stats.trigger_failed += 1
        record = {
            **base_record,
            "status": "trigger_failed",
            "error": str(exc),
            "notes": ["trigger inference failed; verifier skipped"],
        }
        append_jsonl(live_events_path, record)
        print(f"{event_id} status=trigger_failed")
        maybe_remove(tmp_clip_path)
        return

    trigger_status = classify_realtime_decision(trigger_p1, verifier_p1=None, thresholds=loop_cfg.thresholds)
    if trigger_status == "normal_background":
        stats.normal_background += 1
        print(f"{event_id} status=normal_background trigger_p1={trigger_p1:.3f}")
        maybe_remove(tmp_clip_path)
        return

    if trigger_status == "low_confidence_suspicious_motion":
        stats.low_confidence_suspicious_motion += 1
        record = {
            **base_record,
            "status": trigger_status,
            "notes": ["trigger medium; verifier skipped"],
            "trigger_p1": trigger_p1,
            "pose_frame_count": frame_count,
            "pose_detected_frame_count": detected_count,
            "pose_avg_detected_persons": avg_people,
            "pose_note": note,
        }
        append_jsonl(live_events_path, record)
        print(f"{event_id} status={trigger_status} trigger_p1={trigger_p1:.3f}")
        maybe_remove(tmp_clip_path)
        return

    stats.trigger_positive += 1
    saved_clip_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(tmp_clip_path), str(saved_clip_path))

    verifier_p1: float | None = None
    verifier_out: dict[str, Any] | None = None
    try:
        verifier_out = models.verifier.infer_clip(saved_clip_path, event=None, source_type="rtsp_rolling_clip")
        verifier_p1 = verifier_p1_from_output(verifier_out)
        status = classify_realtime_decision(trigger_p1, verifier_p1=verifier_p1, thresholds=loop_cfg.thresholds)
    except Exception as exc:  # noqa: BLE001
        stats.verifier_failed += 1
        status = "verifier_failed"
        verifier_out = {"error": str(exc)}

    if status == "confirmed_suspicious":
        stats.confirmed_suspicious += 1
    elif status == "verifier_rejected":
        stats.verifier_rejected += 1

    record = {
        **base_record,
        "status": status,
        "trigger_status": "trigger_positive",
        "notes": ["trigger high; verifier ran"],
        "clip_path": str(saved_clip_path),
        "trigger_p1": trigger_p1,
        "verifier_p1": verifier_p1,
        "pose_frame_count": frame_count,
        "pose_detected_frame_count": detected_count,
        "pose_avg_detected_persons": avg_people,
        "pose_note": note,
        "trigger_output": trigger_out,
        "verifier_output": verifier_out,
    }
    append_jsonl(live_events_path, record)
    if status == "confirmed_suspicious":
        append_jsonl(alerts_path, record)
    verifier_text = "na" if verifier_p1 is None else f"{verifier_p1:.3f}"
    print(f"{event_id} status={status} trigger_p1={trigger_p1:.3f} verifier_p1={verifier_text}")


def write_summary(path: Path, stats: LiveStats) -> None:
    """Write summary JSON."""
    stats.stopped_at = datetime.now(UTC).isoformat()
    path.write_text(json.dumps(asdict(stats), indent=2), encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    """Run the prototype RTSP rolling loop."""
    setup_logging(args.output_dir, args.debug)
    raw_cfg = load_yaml_config(args.config)
    loop_cfg = build_loop_config(
        raw_cfg,
        trigger_high_threshold=args.trigger_high_threshold,
        verifier_alert_threshold=args.verifier_alert_threshold,
    )
    models = build_runtime_models(raw_cfg)

    live_events_path = args.output_dir / "live_events.jsonl"
    alerts_path = args.output_dir / "alerts.jsonl"
    summary_path = args.output_dir / "summary.json"
    live_events_path.touch(exist_ok=True)
    alerts_path.touch(exist_ok=True)
    stats = LiveStats(started_at=datetime.now(UTC).isoformat())

    masked_url = mask_rtsp_url(args.rtsp_url)
    LOGGER.info("Starting RTSP prototype runner camera=%s stream=%s", args.camera_name, masked_url)
    LOGGER.info(
        "Loop config buffer_seconds=%.1f analysis_interval_seconds=%.1f trigger_high=%.2f verifier_alert=%.2f",
        loop_cfg.buffer_seconds,
        loop_cfg.analysis_interval_seconds,
        loop_cfg.thresholds.trigger_high,
        loop_cfg.thresholds.verifier_alert,
    )

    capture = None
    frame_buffer: deque[tuple[float, Any]] = deque()
    start_monotonic = time.monotonic()
    next_analysis_at = start_monotonic + loop_cfg.analysis_interval_seconds
    sequence_id = 0

    try:
        while True:
            if args.max_seconds is not None and (time.monotonic() - start_monotonic) >= args.max_seconds:
                stats.stop_reason = "max_seconds_reached"
                break

            if capture is None:
                capture = open_capture(args.rtsp_url)
                if capture is None:
                    stats.reconnect_attempts += 1
                    LOGGER.warning("Unable to open RTSP stream %s; retrying", masked_url)
                    time.sleep(loop_cfg.reconnect_delay_seconds)
                    continue
                LOGGER.info("RTSP stream opened for camera=%s", args.camera_name)

            ok, frame_bgr = capture.read()
            if not ok:
                stats.reconnect_attempts += 1
                LOGGER.warning("RTSP frame read failed for %s; reconnecting", masked_url)
                capture.release()
                capture = None
                time.sleep(loop_cfg.reconnect_delay_seconds)
                continue

            import cv2

            now = time.monotonic()
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_buffer.append((now, frame_rgb))
            stats.frames_read += 1

            while frame_buffer and (now - frame_buffer[0][0]) > loop_cfg.buffer_seconds:
                frame_buffer.popleft()

            if now < next_analysis_at:
                continue
            next_analysis_at = now + loop_cfg.analysis_interval_seconds
            if frame_window_duration(frame_buffer) < loop_cfg.min_window_seconds:
                LOGGER.debug("Skipping analysis until rolling buffer reaches minimum duration")
                continue

            sequence_id += 1
            event_id = event_id_for(args.camera_name, sequence_id)
            process_window(
                event_id=event_id,
                camera_name=args.camera_name,
                frames_rgb=[frame for _, frame in frame_buffer],
                output_dir=args.output_dir,
                loop_cfg=loop_cfg,
                models=models,
                live_events_path=live_events_path,
                alerts_path=alerts_path,
                stats=stats,
            )

    except KeyboardInterrupt:
        stats.stop_reason = "keyboard_interrupt"
        LOGGER.info("Ctrl+C received; shutting down RTSP prototype runner")
    finally:
        if capture is not None:
            capture.release()
        write_summary(summary_path, stats)
        LOGGER.info("Summary written to %s", summary_path)
    return 0


def main() -> None:
    """CLI entrypoint."""
    raise SystemExit(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
