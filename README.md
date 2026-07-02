# rerun-lerobot

Convert [Rerun](https://rerun.io) RRD recordings into [LeRobot](https://github.com/huggingface/lerobot) v3 datasets.

`rerun-lerobot` uses the Rerun catalog API to query and transform recordings into the LeRobot v3
format used for imitation-learning training pipelines in PyTorch. The source can be a local
directory of RRD files (served by the OSS Rerun server) or a remote Rerun catalog. It infers data
types from the recordings, resamples all time series to a target frame rate, and writes a LeRobot v3
dataset. Video streams are efficiently remuxed without re-encoding.

## Installation

```bash
pip install rerun-lerobot
```

### ⚠️ Dependency conflict with LeRobot

The conversion relies on the Rerun OSS server API (`rr.server.Server`), which requires
**`rerun-sdk >= 0.27`**. LeRobot currently pins **`rerun-sdk < 0.27`**, so a naive install will fail
to resolve. Override LeRobot's pin at install time.

With `uv`:

```toml
# pyproject.toml
[tool.uv]
override-dependencies = ["rerun-sdk>=0.27"]
```

Or on the command line:

```bash
uv pip install rerun-lerobot --override <(echo "rerun-sdk>=0.27")
```

With `pip`, install and then force the newer `rerun-sdk`:

```bash
pip install rerun-lerobot
pip install --upgrade "rerun-sdk>=0.27"
```

## Usage

The package installs a `rerun-lerobot` CLI that converts recordings into a LeRobot v3 dataset.
Exactly one source is required: a local directory of RRD files (`--rrd-dir`), a remote Rerun
catalog server plus dataset name (`--catalog-url`), or a full Rerun dataset URL (`--dataset-url`).

From a directory of RRD recordings:

```bash
rerun-lerobot \
  --rrd-dir /path/to/recordings \
  --output /path/to/output/dataset \
  --dataset-name my_robot_dataset \
  --fps 10 \
  --index real_time \
  --action /action:Scalars:scalars \
  --state /observation/joint_positions:Scalars:scalars \
  --task /language_instruction:TextDocument:text \
  --video front:/camera/front
```

From a Rerun catalog server (looked up by `--dataset-name`, optional `--catalog-token` for auth):

```bash
rerun-lerobot \
  --catalog-url rerun+http://my-catalog-host:51234 \
  --dataset-name my_robot_dataset \
  --catalog-token "$RERUN_TOKEN" \
  --output /path/to/output/dataset \
  --fps 10 \
  --index real_time \
  --action /action:Scalars:scalars \
  --state /observation/joint_positions:Scalars:scalars \
  --video front:/camera/front
```

Directly from a Rerun dataset URL (bundles the catalog server and dataset id — no `--dataset-name`
needed; `--catalog-token` still applies for auth):

```bash
rerun-lerobot \
  --dataset-url rerun://api.latest-eu.cloud.rerun.io:443/entry/18B40C6FA7631F942c0e90030ac230fa \
  --output /path/to/output/dataset \
  --fps 10 \
  --index real_time \
  --action /action:Scalars:scalars \
  --state /observation/joint_positions:Scalars:scalars \
  --video front:/camera/front
```

### Guided start: discovering columns

You don't need to know the exact column names up front. Start with just a source and an output:

```bash
rerun-lerobot \
  --dataset-url rerun://api.latest-eu.cloud.rerun.io:443/entry/18B40C6FA7631F942c0e90030ac230fa \
  --output /tmp/lerobot
```

Because `--action`, `--state`, and `--fps` are missing, the tool connects to the dataset, prints the
convertible columns it found — action/state candidates (numeric vectors, with dimensions), timelines
for `--index`, task-text candidates, and video streams — and suggests a full command to copy, edit,
and re-run. Pass `--inspect` to do this explicitly without converting.

```
Action / state candidates (numeric vector columns):
  /robot/action:Scalars:scalars          dim 7    [Scalars]
  /observation/joints:Scalars:scalars     dim 6    [Scalars]
  ...
Timelines (for --index):
  log_time      (timestamp[ns])
  ...
Suggested command:
  rerun-lerobot --dataset-url ... \
    --output /tmp/lerobot \
    --fps 10 \
    --index log_time \
    --action /robot/action:Scalars:scalars \
    --state /observation/joints:Scalars:scalars
```

The action/state picks are best-guesses (by name, else the first candidates) — review them: the tool
cannot know which numeric column is the *commanded* action vs the *observed* state.

### Column specification format

Action, state, and task columns are specified as fully qualified columns:

```
entity_path:ComponentName:field_name
```

For example `/robot/action:Scalars:scalars`.

### Video specification format

Videos are specified as `key:path`:

- `key`: camera identifier (e.g. `front`, `wrist`)
- `path`: entity path to the video stream (e.g. `/camera/front`)

The converter expects a [`VideoStream`](https://www.rerun.io/docs/reference/types/archetypes/video_stream)
archetype at the specified paths.

### How video streams are handled when resampling

`--fps` resamples the *scalar* streams (action / state / task): the output rows are the frames of the
chosen `--index` timeline where the action column is present. Video is handled on a **separate path**
— it is not decoded-and-re-timed per output row. There are two modes:

**Default (`--video`, i.e. `use_videos=True`): remux, no re-encoding.**
The original compressed packets from the `VideoStream` (H.264 / HEVC / …) are copied straight into an
MP4 container with their original timestamps — same codec, same frames, no transcoding. This is fast
and lossless. It assumes the source video already runs at (about) the target rate: the converter
compares the source frame rate (median packet interval) against `--fps` and only remuxes if they match
within 5%. **If they differ by more than 5%, conversion errors out** rather than silently resampling —
there is no automatic video re-encode/re-timing yet. In practice, record (or pre-resample) the video
stream at your target `--fps`.

**`--use-images`: decode to raw image frames.**
For each output data row, the frame is decoded at the nearest packet at-or-before that row's timestamp
(latest-at) and stored as an inline image (`dtype: "image"`) instead of a video. This genuinely
resamples the visuals to the output rows, at the cost of decoding every frame and dropping the
compressed video. Use this when the source frame rate does not match `--fps`.

In both modes the frame shape `(height, width, channels)` is inferred by decoding a single frame.

### Full example

```bash
rerun-lerobot \
  --rrd-dir ./robot_recordings \
  --output ./lerobot_dataset \
  --dataset-name robot_demos \
  --fps 15 \
  --action /robot/action:Scalars:scalars \
  --state /robot/state:Scalars:scalars \
  --task /task:TextDocument:text \
  --video front:/camera/front \
  --video wrist:/camera/wrist \
  --action-names "joint_0,joint_1,joint_2,gripper" \
  --state-names "joint_0,joint_1,joint_2,gripper"
```

The output directory contains:

- `data/`: Parquet files with aligned time series data
- `videos/`: encoded video files (unless `--use-images` is passed)
- `meta/`: dataset metadata and episode information

## Python API

```python
from pathlib import Path

from rerun_lerobot import LeRobotConversionConfig, VideoSpec
from rerun_lerobot.lerobot.export import (
    convert_catalog_dataset_to_lerobot,
    convert_dataset_url_to_lerobot,
    convert_rrd_dataset_to_lerobot,
)

config = LeRobotConversionConfig(
    fps=15,
    index_column="real_time",
    action="/robot/action:Scalars:scalars",
    state="/robot/state:Scalars:scalars",
    task="/task:TextDocument:text",
    videos=[VideoSpec(key="front", path="/camera/front")],
)

# From a local directory of RRD files:
convert_rrd_dataset_to_lerobot(
    rrd_dir=Path("./robot_recordings"),
    output_dir=Path("./lerobot_dataset"),
    dataset_name="robot_demos",
    repo_id="robot_demos",
    config=config,
)

# ...or from a remote Rerun catalog:
convert_catalog_dataset_to_lerobot(
    catalog_url="rerun+http://my-catalog-host:51234",
    dataset_name="robot_demos",
    token=None,  # or an auth token
    output_dir=Path("./lerobot_dataset"),
    repo_id="robot_demos",
    config=config,
)

# ...or straight from a Rerun dataset URL:
convert_dataset_url_to_lerobot(
    dataset_url="rerun://api.latest-eu.cloud.rerun.io:443/entry/18B40C6FA7631F942c0e90030ac230fa",
    token=None,  # or an auth token
    output_dir=Path("./lerobot_dataset"),
    repo_id="robot_demos",
    config=config,
)
```

Both delegate to `convert_dataset_to_lerobot(dataset, ...)`, which works on any connected
[`rerun.catalog.DatasetEntry`](https://ref.rerun.io/docs/python/stable/catalog/) if you already
have one.

To discover columns before building a config (the same guidance the CLI prints), use the matching
`inspect_*` function — each returns a `DatasetInspection` you can read or format:

```python
from rerun_lerobot.lerobot.export import inspect_dataset_url

inspection = inspect_dataset_url(
    "rerun://api.latest-eu.cloud.rerun.io:443/entry/18B40C6FA7631F942c0e90030ac230fa",
    token=None,
)
print(inspection.format_report())

for candidate in inspection.action_state_candidates:
    print(candidate.name, candidate.dim, candidate.component)

action_guess, state_guess = inspection.guess_action_and_state()
index_guess = inspection.guess_index()
```

There is also `inspect_catalog_dataset(...)`, `inspect_rrd_dataset(...)`, and
`inspect_dataset(dataset)` for an already-connected `DatasetEntry`.

## Running locally (without publishing to PyPI)

To run the `rerun-lerobot` CLI straight from a checkout of this repo:

```bash
uv sync --dev          # create .venv with the package installed (editable)
uv run rerun-lerobot --help
```

`uv run` executes the entry point from the local source — no build or PyPI upload needed, and edits
to the code take effect immediately. Alternatively, activate the environment and call the binary
directly:

```bash
source .venv/bin/activate
rerun-lerobot --help
```

Or, without cloning, run the latest source from GitHub in a throwaway environment (note the
`rerun-sdk` override, see above):

```bash
uv run --with "git+https://github.com/rerun-io/rerun-lerobot" --with "rerun-sdk>=0.27" \
  --no-project -- rerun-lerobot --help
```

## Development

```bash
uv sync --dev
uv run ruff format --check
uv run ruff check
uv run mypy
uv run pytest
```

The end-to-end test (`tests/test_e2e.py`) downloads a small public RRD sample and
runs a full conversion; it is cached under `tests/data/` and skips automatically
when offline.

## License

Licensed under either of [Apache-2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT) at your option.
