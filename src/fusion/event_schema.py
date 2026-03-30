"""Fusion runtime event schema and serialization helpers."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

EventSource = Literal["frigate_api", "frigate_mqtt", "filesystem", "manifest_csv", "manual"]


@dataclass(slots=True)
class FusionEvent:
    """Normalized event object used by fusion runtime.

    The schema intentionally preserves Frigate-origin fields while supporting
    offline/manual testing inputs.
    """

    event_id: str
    camera_name: str
    timestamp_start: str | float
    timestamp_end: str | float | None = None
    tracked_label: str | None = None
    track_id: str | None = None
    snapshot_path: str | None = None
    clip_path: str | None = None
    recording_path: str | None = None
    pose_input_path: str | None = None
    source: EventSource = "manual"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_frigate_api(cls, payload: dict[str, Any]) -> "FusionEvent":
        """Build from Frigate API event payload."""
        data = payload.get("data", {}) if isinstance(payload.get("data"), dict) else {}
        event_id = str(payload.get("id") or payload.get("event_id") or "unknown_event")
        return cls(
            event_id=event_id,
            camera_name=str(payload.get("camera") or payload.get("camera_name") or "unknown_camera"),
            timestamp_start=payload.get("start_time") or payload.get("start") or payload.get("timestamp_start") or 0.0,
            timestamp_end=payload.get("end_time") or payload.get("end") or payload.get("timestamp_end"),
            tracked_label=payload.get("label") or payload.get("tracked_label"),
            track_id=str(data.get("id") or payload.get("track_id") or "") or None,
            snapshot_path=_str_or_none(payload.get("snapshot_path") or payload.get("snapshot")),
            clip_path=_str_or_none(payload.get("clip_path") or payload.get("clip")),
            recording_path=_str_or_none(payload.get("recording_path") or payload.get("recording")),
            pose_input_path=_str_or_none(payload.get("pose_input_path")),
            source="frigate_api",
            metadata=payload,
        )

    @classmethod
    def from_frigate_mqtt(cls, payload: dict[str, Any]) -> "FusionEvent":
        """Build from Frigate MQTT-like event payload."""
        event = cls.from_frigate_api(payload)
        event.source = "frigate_mqtt"
        return event

    @classmethod
    def from_manifest_row(cls, row: dict[str, Any], source: EventSource = "manifest_csv") -> "FusionEvent":
        """Build from offline CSV/JSON manifest row."""
        metadata = row.get("metadata", {})
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {"raw_metadata": metadata}
        if not isinstance(metadata, dict):
            metadata = {"metadata": metadata}

        return cls(
            event_id=str(row.get("event_id") or row.get("id") or "unknown_event"),
            camera_name=str(row.get("camera_name") or row.get("camera") or "unknown_camera"),
            timestamp_start=row.get("timestamp_start", row.get("start_time", 0.0)),
            timestamp_end=row.get("timestamp_end", row.get("end_time")),
            tracked_label=_str_or_none(row.get("tracked_label") or row.get("label")),
            track_id=_str_or_none(row.get("track_id")),
            snapshot_path=_str_or_none(row.get("snapshot_path")),
            clip_path=_str_or_none(row.get("clip_path")),
            recording_path=_str_or_none(row.get("recording_path")),
            pose_input_path=_str_or_none(row.get("pose_input_path")),
            source=source,
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize event dataclass to a JSON-safe dictionary."""
        return asdict(self)


@dataclass(slots=True)
class FusionDecision:
    """JSON-serializable event-level runtime output."""

    event_id: str
    camera_name: str
    timestamp_start: str | float
    timestamp_end: str | float | None
    source: str
    track_id: str | None
    tracked_label: str | None
    snapshot_path: str | None
    clip_path: str | None
    recording_path: str | None
    pose_input_path: str | None
    trigger_probs: dict[str, float] | None
    trigger_label: int | None
    trigger_confidence: float | None
    verifier_probs: dict[str, float] | None
    verifier_label: int | None
    verifier_confidence: float | None
    final_label: int | None
    final_confidence: float
    decision_stage: str
    notes: list[str]
    review_needed: bool
    status: str

    def keyed_json(self) -> dict[str, dict[str, Any]]:
        """Return JSON payload keyed by event_id and camera_name."""
        return {
            self.event_id: {
                self.camera_name: self.to_dict(),
            }
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize decision dataclass to dict."""
        return asdict(self)


def _str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def load_manifest_csv(path: Path) -> list[FusionEvent]:
    """Load offline events from CSV manifest."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [FusionEvent.from_manifest_row(row, source="manifest_csv") for row in reader]


def load_json_events(path: Path, source: EventSource = "manual") -> list[FusionEvent]:
    """Load one or many events from JSON payload file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        if "events" in payload and isinstance(payload["events"], list):
            rows: Iterable[dict[str, Any]] = payload["events"]
        else:
            rows = [payload]
    elif isinstance(payload, list):
        rows = payload
    else:
        raise ValueError(f"Unsupported JSON payload in {path}")
    return [FusionEvent.from_manifest_row(row, source=source) for row in rows]
