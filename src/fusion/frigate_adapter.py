"""Frigate integration adapters for fusion runtime."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src.fusion.event_schema import FusionEvent

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class FrigateAdapterConfig:
    """Config for Frigate API interactions and path heuristics."""

    api_base_url: str | None = None
    api_key: str | None = None
    clips_root: Path = Path("/media/frigate/clips")
    recordings_root: Path = Path("/media/frigate/recordings")


class FrigateAdapter:
    """Adapter that converts Frigate-origin data into FusionEvent objects."""

    def __init__(self, config: FrigateAdapterConfig) -> None:
        self.config = config

    def from_api_event(self, payload: dict[str, Any]) -> FusionEvent:
        """Map Frigate HTTP event payload to FusionEvent."""
        event = FusionEvent.from_frigate_api(payload)
        return self.resolve_media_paths(event)

    def from_mqtt_payload(self, payload: dict[str, Any]) -> FusionEvent:
        """Map Frigate MQTT-like payload to FusionEvent."""
        event = FusionEvent.from_frigate_mqtt(payload)
        return self.resolve_media_paths(event)

    def from_manifest_row(self, row: dict[str, Any]) -> FusionEvent:
        """Map CSV/JSON manifest row to FusionEvent."""
        event = FusionEvent.from_manifest_row(row, source="manifest_csv")
        return self.resolve_media_paths(event)

    def resolve_event(self, event_id: str) -> FusionEvent:
        """Resolve event by Frigate event ID via API then path heuristics."""
        payload = self.fetch_event_from_api(event_id)
        return self.from_api_event(payload)

    def fetch_event_from_api(self, event_id: str) -> dict[str, Any]:
        """Fetch raw Frigate event payload from API."""
        if not self.config.api_base_url:
            raise ValueError("Frigate api_base_url is required to resolve event IDs")
        base = self.config.api_base_url.rstrip("/")
        candidates = [
            f"{base}/api/events/{event_id}",
            f"{base}/api/event/{event_id}",
        ]
        headers = {"Accept": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        last_error: Exception | None = None
        for url in candidates:
            try:
                request = Request(url=url, headers=headers, method="GET")
                with urlopen(request, timeout=15) as response:
                    body = response.read().decode("utf-8")
                    payload = json.loads(body)
                LOGGER.debug("Fetched Frigate event %s from %s", event_id, url)
                return payload
            except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
                LOGGER.debug("Frigate fetch candidate failed for %s: %s", url, exc)
        raise RuntimeError(f"Failed to fetch Frigate event_id={event_id}: {last_error}")

    def resolve_media_paths(self, event: FusionEvent) -> FusionEvent:
        """Best-effort resolver for clip/snapshot/recording paths."""
        if event.clip_path:
            clip_path = Path(event.clip_path)
            if not clip_path.exists():
                LOGGER.debug("clip_path set but missing on disk: %s", clip_path)

        if not event.clip_path:
            clip_candidates = [
                self.config.clips_root / f"{event.camera_name}-{event.event_id}.mp4",
                self.config.clips_root / f"{event.event_id}.mp4",
            ]
            for candidate in clip_candidates:
                if candidate.exists():
                    event.clip_path = str(candidate)
                    break

        if not event.snapshot_path:
            snapshot_candidates = [
                self.config.clips_root / f"{event.camera_name}-{event.event_id}.jpg",
                self.config.clips_root / f"{event.event_id}.jpg",
            ]
            for candidate in snapshot_candidates:
                if candidate.exists():
                    event.snapshot_path = str(candidate)
                    break

        if not event.recording_path:
            camera_recordings_dir = self.config.recordings_root / event.camera_name
            if camera_recordings_dir.exists():
                event.recording_path = str(camera_recordings_dir)

        event.metadata = {
            **event.metadata,
            "resolver": {
                "clips_root": str(self.config.clips_root),
                "recordings_root": str(self.config.recordings_root),
            },
        }
        return event


def parse_mqtt_json(payload_text: str) -> FusionEvent:
    """Parse raw MQTT JSON text into FusionEvent."""
    payload = json.loads(payload_text)
    return FusionEvent.from_frigate_mqtt(payload)
