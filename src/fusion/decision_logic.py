"""Two-stage fusion decision runtime for Trigger + Verifier inference."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from src.data.video_preprocess import FrigateEvent as VerifierEvent
from src.data.video_preprocess import VideoProcessConfig, frigate_event_to_clip
from src.fusion.event_schema import FusionDecision, FusionEvent
from src.trigger.infer import FrigateEvent as TriggerEvent

LOGGER = logging.getLogger(__name__)


class PoseExtractionHook(Protocol):
    """Optional hook to derive pose path from event media."""

    def __call__(self, event: FusionEvent) -> str | None:
        """Extract or resolve pose file and return a path when available."""


@dataclass(slots=True)
class FusionThresholds:
    """Threshold config for Trigger/Verifier fusion behavior."""

    trigger_escalation_threshold: float = 0.55
    trigger_force_verify_threshold: float = 0.75
    trigger_normal_drop_threshold: float = 0.90
    verifier_fight_threshold: float = 0.65
    verifier_tension_threshold: float = 0.55
    verifier_normal_override_threshold: float = 0.75
    escalate_uncertain_trigger: bool = True
    review_needed_on_missing_verifier: bool = True


@dataclass(slots=True)
class FusionRuntimeConfig:
    """Runtime knobs for batch operation and media handling."""

    extraction_output_dir: Path
    video_process: VideoProcessConfig
    thresholds: FusionThresholds
    drop_trigger_normal_early: bool = True


@dataclass(slots=True)
class RuntimeStats:
    """Runtime counters used for summary.json."""

    processed: int = 0
    dropped_at_trigger: int = 0
    escalated_to_verifier: int = 0
    finalized_0: int = 0
    finalized_1: int = 0
    finalized_2: int = 0
    missing_media: int = 0
    review_needed: int = 0


class FusionRuntime:
    """Production-friendly fusion runtime orchestrator."""

    def __init__(
        self,
        trigger_inferencer: Any,
        verifier_inferencer: Any,
        config: FusionRuntimeConfig,
        pose_extraction_hook: PoseExtractionHook | None = None,
    ) -> None:
        self.trigger_inferencer = trigger_inferencer
        self.verifier_inferencer = verifier_inferencer
        self.config = config
        self.pose_extraction_hook = pose_extraction_hook

    def process_event(self, event: FusionEvent, dry_run: bool = False) -> FusionDecision:
        """Run full dual-model logic for one event."""
        notes: list[str] = []
        trigger_out: dict[str, Any] | None = None
        verifier_out: dict[str, Any] | None = None

        if not event.pose_input_path and self.pose_extraction_hook and (event.clip_path or event.recording_path):
            extracted_pose = self.pose_extraction_hook(event)
            if extracted_pose:
                event.pose_input_path = extracted_pose
                notes.append("pose extracted from video fallback hook")

        if not event.pose_input_path and not (event.clip_path or event.recording_path):
            notes.append("missing pose and usable video media")
            return self._fallback_decision(event, notes, status="failed", review_needed=True)

        if dry_run:
            notes.append("dry-run enabled; models not executed")
            decision = self._fallback_decision(event, notes, status="fallback", review_needed=False)
            decision.decision_stage = "dry_run"
            return decision

        if event.pose_input_path:
            trigger_event = TriggerEvent(
                event_id=event.event_id,
                camera_name=event.camera_name,
                timestamp_start=event.timestamp_start,
                timestamp_end=event.timestamp_end,
                tracked_label=event.tracked_label,
                track_id=event.track_id,
                snapshot_path=event.snapshot_path,
                clip_path=event.clip_path,
                recording_path=event.recording_path,
                pose_input_path=event.pose_input_path,
                source=event.source,
                metadata=event.metadata,
            )
            trigger_out = self.trigger_inferencer.infer_frigate_event(trigger_event)
            notes.extend(self._trigger_notes(trigger_out))
        else:
            notes.append("trigger skipped because pose input unavailable")

        escalate = self._should_escalate(trigger_out)
        if trigger_out is None:
            escalate = True
            notes.append("escalating without trigger output")

        if not escalate:
            notes.append("event dropped at trigger stage")
            return self._build_decision(
                event=event,
                trigger_out=trigger_out,
                verifier_out=None,
                final_label=0,
                final_confidence=float((trigger_out or {}).get("class_probabilities", {}).get("0", 0.0)),
                decision_stage="trigger",
                notes=notes,
                review_needed=False,
                status="trigger_only",
            )

        notes.append("trigger escalation path selected")
        verifier_clip_path: Path | None = None
        source_type = "unknown"
        if event.clip_path or event.recording_path:
            verifier_event = VerifierEvent(
                event_id=event.event_id,
                camera_name=event.camera_name,
                timestamp_start=event.timestamp_start,
                timestamp_end=event.timestamp_end,
                tracked_label=event.tracked_label,
                track_id=event.track_id,
                snapshot_path=event.snapshot_path,
                clip_path=event.clip_path,
                recording_path=event.recording_path,
                source=event.source,
                metadata=event.metadata,
            )
            verifier_clip_path, source_type = frigate_event_to_clip(
                verifier_event,
                output_dir=self.config.extraction_output_dir,
                process_cfg=self.config.video_process,
            )

        if verifier_clip_path is None:
            if event.snapshot_path and not (event.clip_path or event.recording_path):
                notes.append("verifier unavailable because only snapshot was present")
            else:
                notes.append("verifier unavailable due to missing clip/recording")
            review_needed = self.config.thresholds.review_needed_on_missing_verifier
            status = "review_needed" if review_needed else "fallback"
            return self._build_decision(
                event=event,
                trigger_out=trigger_out,
                verifier_out=None,
                final_label=(trigger_out or {}).get("predicted_label"),
                final_confidence=float((trigger_out or {}).get("confidence") or 0.0),
                decision_stage="trigger",
                notes=notes,
                review_needed=review_needed,
                status=status,
            )

        if source_type == "frigate_recording_extract":
            notes.append("clip extracted from recording for verifier")

        verifier_out = self.verifier_inferencer.infer_clip(
            verifier_clip_path,
            event=verifier_event,
            source_type=source_type,
        )
        notes.extend(self._verifier_notes(verifier_out))
        final_label, final_confidence = self._fuse_labels(trigger_out, verifier_out, notes)
        return self._build_decision(
            event=event,
            trigger_out=trigger_out,
            verifier_out=verifier_out,
            final_label=final_label,
            final_confidence=final_confidence,
            decision_stage="verifier",
            notes=notes,
            review_needed=False,
            status="ok",
        )

    def _should_escalate(self, trigger_out: dict[str, Any] | None) -> bool:
        if trigger_out is None:
            return True
        probs = trigger_out.get("class_probabilities", {})
        p0 = float(probs.get("0", 0.0))
        p1 = float(probs.get("1", 0.0))
        p2 = float(probs.get("2", 0.0))
        pred = int(trigger_out.get("predicted_label", 0))
        confidence = float(trigger_out.get("confidence", 0.0))
        th = self.config.thresholds

        if self.config.drop_trigger_normal_early and pred == 0 and p0 >= th.trigger_normal_drop_threshold:
            return False
        if max(p1, p2) >= th.trigger_force_verify_threshold:
            return True
        if pred in (1, 2) and confidence >= th.trigger_escalation_threshold:
            return True
        return th.escalate_uncertain_trigger

    def _fuse_labels(
        self,
        trigger_out: dict[str, Any] | None,
        verifier_out: dict[str, Any],
        notes: list[str],
    ) -> tuple[int, float]:
        th = self.config.thresholds
        v_probs = verifier_out.get("class_probabilities", {})
        v0 = float(v_probs.get("0", 0.0))
        v1 = float(v_probs.get("1", 0.0))
        v2 = float(v_probs.get("2", 0.0))

        if v0 >= th.verifier_normal_override_threshold:
            notes.append("verifier downgraded event to normal")
            return 0, v0
        if v2 >= th.verifier_fight_threshold:
            notes.append("verifier confirmed fight")
            return 2, v2
        if v1 >= th.verifier_tension_threshold:
            notes.append("verifier confirmed pre-fight tension")
            return 1, v1

        verifier_label = int(verifier_out.get("predicted_label", 0))
        verifier_conf = float(verifier_out.get("confidence", 0.0))
        if verifier_label == 2:
            notes.append("verifier predicted fight below strict threshold; accepted with lower confidence")
            return 2, verifier_conf
        if verifier_label == 1:
            notes.append("verifier predicted pre-fight tension")
            return 1, verifier_conf

        if trigger_out is not None:
            notes.append("verifier ambiguous; preserving trigger output")
            return int(trigger_out.get("predicted_label", 0)), float(trigger_out.get("confidence", 0.0))
        notes.append("verifier only decision path")
        return verifier_label, verifier_conf

    def _build_decision(
        self,
        event: FusionEvent,
        trigger_out: dict[str, Any] | None,
        verifier_out: dict[str, Any] | None,
        final_label: int | None,
        final_confidence: float,
        decision_stage: str,
        notes: list[str],
        review_needed: bool,
        status: str,
    ) -> FusionDecision:
        return FusionDecision(
            event_id=event.event_id,
            camera_name=event.camera_name,
            timestamp_start=event.timestamp_start,
            timestamp_end=event.timestamp_end,
            source=event.source,
            track_id=event.track_id,
            tracked_label=event.tracked_label,
            snapshot_path=event.snapshot_path,
            clip_path=event.clip_path,
            recording_path=event.recording_path,
            pose_input_path=event.pose_input_path,
            trigger_probs=(trigger_out or {}).get("class_probabilities"),
            trigger_label=(trigger_out or {}).get("predicted_label"),
            trigger_confidence=(trigger_out or {}).get("confidence"),
            verifier_probs=(verifier_out or {}).get("class_probabilities"),
            verifier_label=(verifier_out or {}).get("predicted_label"),
            verifier_confidence=(verifier_out or {}).get("confidence"),
            final_label=final_label,
            final_confidence=final_confidence,
            decision_stage=decision_stage,
            notes=notes,
            review_needed=review_needed,
            status=status,
        )

    def _fallback_decision(self, event: FusionEvent, notes: list[str], status: str, review_needed: bool) -> FusionDecision:
        return self._build_decision(
            event=event,
            trigger_out=None,
            verifier_out=None,
            final_label=None,
            final_confidence=0.0,
            decision_stage="fallback",
            notes=notes,
            review_needed=review_needed,
            status=status,
        )

    @staticmethod
    def _trigger_notes(trigger_out: dict[str, Any]) -> list[str]:
        pred = int(trigger_out.get("predicted_label", 0))
        conf = float(trigger_out.get("confidence", 0.0))
        if pred == 0 and conf >= 0.85:
            return ["trigger predicted normal with high confidence"]
        if pred == 1:
            return ["trigger flagged pre-fight posture/tension"]
        if pred == 2:
            return ["trigger flagged aggressive posture"]
        return ["trigger output collected"]

    @staticmethod
    def _verifier_notes(verifier_out: dict[str, Any]) -> list[str]:
        pred = verifier_out.get("predicted_label")
        if pred == 0:
            return ["verifier reviewed clip as normal"]
        if pred == 1:
            return ["verifier reviewed clip as pre-fight/tension"]
        if pred == 2:
            return ["verifier reviewed clip as fight"]
        return ["verifier output unavailable"]


def update_stats(stats: RuntimeStats, decision: FusionDecision) -> RuntimeStats:
    """Mutate and return runtime summary counters."""
    stats.processed += 1
    if decision.decision_stage == "trigger" and decision.status == "trigger_only":
        stats.dropped_at_trigger += 1
    if decision.decision_stage == "verifier":
        stats.escalated_to_verifier += 1
    if decision.final_label == 0:
        stats.finalized_0 += 1
    elif decision.final_label == 1:
        stats.finalized_1 += 1
    elif decision.final_label == 2:
        stats.finalized_2 += 1
    if decision.status in {"failed", "fallback", "review_needed"}:
        stats.missing_media += 1
    if decision.review_needed:
        stats.review_needed += 1
    return stats
