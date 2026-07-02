"""Tests for schema inspection / candidate classification (pyarrow only, no rerun-sdk)."""

from __future__ import annotations

import pyarrow as pa

from rerun_lerobot.inspection import classify_schema, suggest_command


def _sample_schema() -> pa.Schema:
    """A schema resembling a Rerun dataset reader schema."""
    return pa.schema([
        # Timelines (no "entity_path:Component:field" structure).
        ("log_time", pa.timestamp("ns")),
        ("example_time", pa.duration("ns")),
        ("log_tick", pa.int64()),
        ("rerun.controls.RowId", pa.binary(16)),  # control column, ignored
        # Numeric vectors -> action/state candidates.
        ("/robot/action:Scalars:scalars", pa.list_(pa.list_(pa.float64(), 4))),
        ("/observation/joints:Scalars:scalars", pa.list_(pa.list_(pa.float32(), 6))),
        ("/transforms:Transform3D:translation", pa.list_(pa.list_(pa.float32(), 3))),
        # Asset payloads / metadata -> excluded from action/state.
        ("/mesh:Asset3D:blob", pa.list_(pa.list_(pa.uint8()))),
        ("property:RecordingInfo:start_time", pa.timestamp("ns")),
        # Text -> task candidate (only Text/TextDocument components).
        ("/instruction:TextDocument:text", pa.list_(pa.string())),
        ("/mesh:Asset3D:media_type", pa.list_(pa.string())),  # string but not a task
        # Video stream sample -> video candidate.
        ("/camera/front:VideoStream:sample", pa.list_(pa.binary())),
    ])


def test_classify_schema_action_state() -> None:
    insp = classify_schema(_sample_schema(), dataset_name="demo", num_segments=3)

    assert insp.dataset_name == "demo"
    assert insp.num_segments == 3

    names = {c.name for c in insp.action_state_candidates}
    assert names == {
        "/robot/action:Scalars:scalars",
        "/observation/joints:Scalars:scalars",
        "/transforms:Transform3D:translation",
    }
    # Asset blob and static property are excluded.
    assert not any("Asset3D" in n for n in names)
    assert not any(n.startswith("property:") for n in names)

    dims = {c.name: c.dim for c in insp.action_state_candidates}
    assert dims["/robot/action:Scalars:scalars"] == 4
    assert dims["/transforms:Transform3D:translation"] == 3


def test_classify_schema_timelines_task_video() -> None:
    insp = classify_schema(_sample_schema())

    timeline_names = [t.name for t in insp.timelines]
    assert "log_time" in timeline_names
    assert "example_time" in timeline_names
    assert "log_tick" in timeline_names
    assert "rerun.controls.RowId" not in timeline_names  # control column filtered

    # Temporal timelines sort ahead of non-temporal ones.
    assert insp.timelines[0].is_temporal
    assert {t.name for t in insp.timelines if not t.is_temporal} == {"log_tick"}

    assert [c.name for c in insp.task_candidates] == ["/instruction:TextDocument:text"]
    assert insp.video_paths == ["/camera/front"]


def test_guess_index_prefers_temporal() -> None:
    insp = classify_schema(_sample_schema())
    assert insp.guess_index() == "example_time"  # first temporal after sort


def test_guess_action_and_state_uses_names() -> None:
    insp = classify_schema(_sample_schema())
    action, state = insp.guess_action_and_state()
    assert action == "/robot/action:Scalars:scalars"
    # "observation"/"joints" beats the transform for state.
    assert state == "/observation/joints:Scalars:scalars"


def test_suggest_command_contains_picks() -> None:
    insp = classify_schema(_sample_schema())
    command = suggest_command(insp, source_args=["--dataset-url", "rerun://h:443/entry/abc"], output="/tmp/out")
    assert "rerun-lerobot --dataset-url rerun://h:443/entry/abc" in command
    assert "--output /tmp/out" in command
    assert "--index example_time" in command
    assert "--action /robot/action:Scalars:scalars" in command
    assert "--state /observation/joints:Scalars:scalars" in command


def test_report_truncates_long_candidate_list() -> None:
    fields = [("log_time", pa.timestamp("ns"))]
    fields += [(f"/j{i}:Scalars:scalars", pa.list_(pa.float32())) for i in range(40)]
    insp = classify_schema(pa.schema(fields))
    assert len(insp.action_state_candidates) == 40
    report = insp.format_report()
    assert "and 10 more" in report
