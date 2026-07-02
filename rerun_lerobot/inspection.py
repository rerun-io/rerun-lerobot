"""
Schema inspection: discover convertible columns in a Rerun dataset.

Given a dataset's Arrow schema, classify its columns into the pieces a LeRobot
conversion needs — timelines (``--index``), numeric vectors (``--action`` /
``--state``), task text (``--task``), and video streams (``--video``) — so the
user can be guided to a working command instead of having to know the exact
``entity_path:Component:field`` spelling up front.

This module depends only on ``pyarrow`` so it stays cheap to import and easy to
unit-test with a hand-built schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pyarrow as pa

if TYPE_CHECKING:
    from collections.abc import Sequence

# Components that are numeric but are never a robot action/state (asset payloads,
# colors, coordinate-frame bookkeeping, recording metadata). Excluded from the
# action/state candidate list to cut noise.
_NON_SIGNAL_COMPONENTS = frozenset({"Asset3D", "CoordinateFrame", "RecordingInfo"})

# Components that carry a natural-language task/instruction.
_TASK_COMPONENTS = frozenset({"Text", "TextDocument"})

_VIDEO_SAMPLE_SUFFIX = ":VideoStream:sample"

# Max action/state candidates to print before truncating the report.
_MAX_LISTED_CANDIDATES = 30


@dataclass(frozen=True)
class ColumnCandidate:
    """A column that can be mapped to a LeRobot feature."""

    name: str  # fully qualified "entity_path:Component:field"
    component: str
    dim: int | None  # fixed vector length if known, else None (variable)


@dataclass(frozen=True)
class TimelineCandidate:
    """A timeline that can be used as the conversion index."""

    name: str
    arrow_type: str
    is_temporal: bool  # timestamp/duration (vs. a plain sequence like log_tick)


@dataclass
class DatasetInspection:
    """The result of inspecting a dataset schema: everything needed to guide a conversion."""

    dataset_name: str | None = None
    num_segments: int | None = None
    timelines: list[TimelineCandidate] = field(default_factory=list)
    action_state_candidates: list[ColumnCandidate] = field(default_factory=list)
    task_candidates: list[ColumnCandidate] = field(default_factory=list)
    video_paths: list[str] = field(default_factory=list)

    def guess_index(self) -> str | None:
        """Best-guess timeline for ``--index``: prefer a temporal one, else the first."""
        for timeline in self.timelines:
            if timeline.is_temporal:
                return timeline.name
        return self.timelines[0].name if self.timelines else None

    def guess_action_and_state(self) -> tuple[str | None, str | None]:
        """
        Best-guess (action, state) columns from naming conventions.

        Falls back to the first two numeric candidates when names give no hint.
        """
        action: str | None = None
        state: str | None = None
        for candidate in self.action_state_candidates:
            lowered = candidate.name.lower()
            if action is None and "action" in lowered:
                action = candidate.name
            elif state is None and any(key in lowered for key in ("state", "observation", "joint", "position")):
                state = candidate.name

        remaining = [c.name for c in self.action_state_candidates if c.name not in (action, state)]
        if action is None and remaining:
            action = remaining.pop(0)
        if state is None and remaining:
            state = remaining.pop(0)
        return action, state

    def format_report(self, *, suggested_command: str | None = None) -> str:
        """Render a human-readable guidance report."""
        lines: list[str] = []
        header = "Inspected dataset"
        if self.dataset_name:
            header += f" '{self.dataset_name}'"
        if self.num_segments is not None:
            header += f" ({self.num_segments} segment{'s' if self.num_segments != 1 else ''})"
        lines.append(header)
        lines.append("")

        lines.append("Action / state candidates (numeric vector columns):")
        if self.action_state_candidates:
            shown = self.action_state_candidates[:_MAX_LISTED_CANDIDATES]
            width = max(len(c.name) for c in shown)
            for candidate in shown:
                dim = f"dim {candidate.dim}" if candidate.dim is not None else "dim ?"
                lines.append(f"  {candidate.name:<{width}}  {dim:<7}  [{candidate.component}]")
            hidden = len(self.action_state_candidates) - len(shown)
            if hidden > 0:
                lines.append(f"  ... and {hidden} more (full list available via the Python API)")
        else:
            lines.append("  (none found)")
        lines.append("")

        lines.append("Timelines (for --index):")
        if self.timelines:
            width = max(len(t.name) for t in self.timelines)
            for timeline in self.timelines:
                marker = "" if timeline.is_temporal else "  (non-temporal)"
                lines.append(f"  {timeline.name:<{width}}  ({timeline.arrow_type}){marker}")
        else:
            lines.append("  (none found)")
        lines.append("")

        lines.append("Task text candidates (optional, for --task):")
        if self.task_candidates:
            for candidate in self.task_candidates:
                lines.append(f"  {candidate.name}  [{candidate.component}]")
        else:
            lines.append("  (none found)")
        lines.append("")

        lines.append("Video streams (optional, for --video key:path):")
        if self.video_paths:
            for path in self.video_paths:
                key = path.rstrip("/").split("/")[-1] or "cam"
                lines.append(f"  {key}:{path}")
        else:
            lines.append("  (none found)")

        if suggested_command is not None:
            lines.append("")
            lines.append("Suggested command:")
            lines.append(suggested_command)

        return "\n".join(lines)


def _unwrap_list_type(arrow_type: pa.DataType) -> tuple[pa.DataType, int | None]:
    """Peel list layers off an Arrow type, returning (leaf_type, fixed_dim_or_None)."""
    dim: int | None = None
    current = arrow_type
    while pa.types.is_list(current) or pa.types.is_large_list(current) or pa.types.is_fixed_size_list(current):
        if pa.types.is_fixed_size_list(current):
            dim = current.list_size
        current = current.value_type
    return current, dim


def _is_control_column(name: str) -> bool:
    """True for internal Rerun bookkeeping columns that are never a user timeline."""
    return name.startswith("rerun.") or name.startswith("rerun_")


def classify_schema(
    schema: pa.Schema,
    *,
    dataset_name: str | None = None,
    num_segments: int | None = None,
) -> DatasetInspection:
    """Classify an Arrow schema into timeline / action-state / task / video candidates."""
    inspection = DatasetInspection(dataset_name=dataset_name, num_segments=num_segments)

    for name in schema.names:
        arrow_type = schema.field(name).type

        # Static properties (e.g. "property:RecordingInfo:start_time") are metadata, not signals.
        if name.startswith("property:") or name.startswith("property/"):
            continue

        # Timeline columns have no "entity_path:Component:field" structure.
        if ":" not in name:
            if _is_control_column(name):
                continue
            is_temporal = pa.types.is_timestamp(arrow_type) or pa.types.is_duration(arrow_type)
            is_sequence = pa.types.is_integer(arrow_type)
            if is_temporal or is_sequence:
                inspection.timelines.append(
                    TimelineCandidate(name=name, arrow_type=str(arrow_type), is_temporal=is_temporal)
                )
            continue

        parts = name.split(":")
        component = parts[1] if len(parts) >= 2 else ""

        if name.endswith(_VIDEO_SAMPLE_SUFFIX):
            inspection.video_paths.append(parts[0])
            continue

        leaf, dim = _unwrap_list_type(arrow_type)

        if pa.types.is_string(leaf) or pa.types.is_large_string(leaf):
            if component in _TASK_COMPONENTS:
                inspection.task_candidates.append(ColumnCandidate(name=name, component=component, dim=dim))
            continue

        if (pa.types.is_floating(leaf) or pa.types.is_integer(leaf)) and component not in _NON_SIGNAL_COMPONENTS:
            inspection.action_state_candidates.append(ColumnCandidate(name=name, component=component, dim=dim))

    # Fixed, small vectors first (most action/state-like), then variable/large, then by name.
    inspection.action_state_candidates.sort(key=lambda c: (c.dim is None, c.dim if c.dim is not None else 0, c.name))
    inspection.timelines.sort(key=lambda t: (not t.is_temporal, t.name))
    inspection.task_candidates.sort(key=lambda c: c.name)
    inspection.video_paths.sort()

    return inspection


def suggest_command(
    inspection: DatasetInspection,
    *,
    source_args: Sequence[str],
    output: str,
    default_fps: int = 10,
) -> str:
    """
    Build a ready-to-edit ``rerun-lerobot`` command from best-guess picks.

    Args:
        inspection: The classified schema.
        source_args: The source selector already known (e.g. ``["--dataset-url", url]``).
        output: The output directory path.
        default_fps: FPS placeholder to suggest (the tool cannot infer it reliably).

    """
    action, state = inspection.guess_action_and_state()
    index = inspection.guess_index()
    placeholder = "<ENTITY_PATH:Component:field>"

    # Each entry is one logical piece of the command, rendered on its own line.
    segments: list[str] = ["rerun-lerobot " + " ".join(source_args), f"--output {output}", f"--fps {default_fps}"]
    if index is not None:
        segments.append(f"--index {index}")
    segments.append(f"--action {action if action is not None else placeholder}")
    segments.append(f"--state {state if state is not None else placeholder}")

    return "  " + " \\\n    ".join(segments)
