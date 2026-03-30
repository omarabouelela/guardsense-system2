"""Shared label definitions for GuardSense System 2."""

from __future__ import annotations

from dataclasses import dataclass

LABEL_NORMAL = 0
LABEL_PRE_FIGHT = 1
LABEL_FIGHT = 2

LABEL_TO_NAME: dict[int, str] = {
    LABEL_NORMAL: "normal",
    LABEL_PRE_FIGHT: "pre_fight_tension",
    LABEL_FIGHT: "fight",
}
NAME_TO_LABEL: dict[str, int] = {v: k for k, v in LABEL_TO_NAME.items()}


@dataclass(frozen=True, slots=True)
class LabelSpec:
    """Normalized class metadata."""

    class_id: int
    name: str
    description: str


LABEL_SPECS: tuple[LabelSpec, ...] = (
    LabelSpec(0, "normal", "Normal / hard-negative / harmless but visually confusing activity."),
    LabelSpec(1, "pre_fight_tension", "Escalation, tension, and pre-fight behavior before physical violence."),
    LabelSpec(2, "fight", "Active physical fighting and violent contact."),
)
