# Rerun API wishlist

Friction hit while building `rerun-lerobot` against `rerun-sdk` 0.33. Each item lists what I
wanted, the workaround I used, and where it lives in this repo. Ordered roughly by how much pain
they caused.

## 1. List a dataset's timelines / indexes

**Want:** `dataset.timelines()` (or `.indexes()`) returning each timeline's name, kind
(time vs. sequence), and element type — e.g. `[{name: "log_time", kind: "time", type: "timestamp[ns]"}, ...]`.

**Workaround:** read `dataset.arrow_schema()` and treat every column *without* a
`entity_path:Component:field` structure as a timeline candidate, dropping control columns
(`rerun.*`, `rerun_segment_id`) and guessing "temporal" from the Arrow type being timestamp/duration.
See `classify_schema()` in `rerun_lerobot/inspection.py`.

This is the big one — the user explicitly asked "can `--action`/`--state`/`--index` be inferred",
and choosing an index means enumerating timelines, which has no first-class API today.

## 2. Structured column/component listing (no string parsing)

**Want:** a way to enumerate a dataset's columns as structured records: entity path, archetype,
component, field, whether it's static, element dtype, and fixed dimensionality. Something like
`dataset.columns()` → `[ColumnInfo(entity_path=..., component=..., field=..., element_type=..., dim=...)]`.

**Workaround:** parse the `"entity_path:Component:field"` column-name strings by hand and walk the
Arrow type (peeling `list`/`fixed_size_list` layers) to find the leaf dtype and vector dimension.
See `_unwrap_list_type()` and `classify_schema()` in `rerun_lerobot/inspection.py`.

## 3. Open a `DatasetEntry` straight from a dataset URL

**Want:** `rr.catalog.dataset_from_url("rerun://host:443/entry/<id>", token=...)` (or
`CatalogClient.from_dataset_url(...)`) returning a connected `DatasetEntry`.

**Workaround:** manually split `rerun://host:port/entry/<id>` into `(origin, entry_id)`, construct
`CatalogClient(origin, token=...)`, then `get_dataset(id=entry_id)`.
See `split_dataset_url()` in `rerun_lerobot/utils.py` and `convert_dataset_url_to_lerobot()` /
`inspect_dataset_url()` in `rerun_lerobot/lerobot/export.py`.

## 4. Index range / stats for a timeline

**Want:** `dataset.index_range(timeline)` → `(min, max, count)` (or per-segment), cheaply, without
reading the whole column.

**Why:** to *infer or suggest an FPS* and to build a resampling grid. Right now the tool cannot
suggest a sensible `--fps` — it prints a placeholder (`10`) — because there's no cheap way to learn
a timeline's span and cadence. See `suggest_command(default_fps=...)` in `rerun_lerobot/inspection.py`.

## 5. First-class resample-to-fixed-FPS

**Want:** a documented, worked example (or helper) for "resample all rows to N Hz on timeline T",
building on `reader(using_index_values=...)` / `IndexValuesLike` + `fill_latest_at`.

**Why:** `--fps` currently only sets LeRobot metadata; `total_frames` equals the raw row count
(verified in the e2e test — 60k rows in, 60k frames out at `--fps 10`). Wiring true resampling needs
an index grid from #4 plus a clear latest-at fill story. See `convert_dataset_to_lerobot()` in
`rerun_lerobot/lerobot/export.py`.

## 6. Typed video-stream discovery

**Want:** `dataset.video_streams()` → paths with codec + resolution, without decoding.

**Workaround:** detect video by the `:VideoStream:sample` column-name suffix, and *decode one frame*
just to learn `(height, width, channels)`. See `infer_video_shape_from_table()` in
`rerun_lerobot/lerobot/video_processing.py` and the video branch of `classify_schema()`.

## 7. Cheap "sample one value per column"

**Want:** `reader(...).limit(n)` or a "head" that returns a few rows, for feature inference.

**Workaround:** query a whole segment into a PyArrow table and take the first non-null per column.
See `infer_features()` in `rerun_lerobot/lerobot/feature_inference.py` and the inference query in
`convert_dataset_to_lerobot()`.

## 8. Per-segment emptiness / size on the segment listing

**Want:** `segment_ids()` (or a sibling) to expose size / emptiness per segment, e.g.
`[{id, size_bytes, is_empty}]`.

**Workaround:** for every segment, pull `dataset.segment_table()` and scan its rows matching
`rerun_segment_id` to read `rerun_size_bytes` and skip 0-byte segments. See the per-segment loop in
`convert_dataset_to_lerobot()` in `rerun_lerobot/lerobot/export.py`.

## 9. Static-property columns are noise in discovery

**Want:** the schema to mark static/property columns (or an option to exclude them), so
`property:RecordingInfo:start_time` and friends don't show up as convertible-signal candidates.

**Workaround:** drop any column whose name starts with `property:`. See `classify_schema()` in
`rerun_lerobot/inspection.py`.

## Minor / papercuts

- **`Server.url` is a method, not a property.** `print(server.url)` yields `<bound method ...>`;
  you must call `server.url()`. A property (or a clear docstring) would remove a common footgun.
  Used in `tests/test_e2e.py`.
- **`rerun-sdk` version vs. LeRobot.** The OSS server / catalog API needs `rerun-sdk >= 0.27`, but
  LeRobot pins `< 0.27`, so every install needs a dependency override. Documented in `README.md` and
  `pyproject.toml` (`[tool.uv] override-dependencies`). Not a Rerun API gap per se, but painful for
  downstream packaging.
- **`datafusion` major-version coupling.** A plain `datafusion` dependency resolves to a version the
  catalog client rejects (`RerunIncompatibleDependencyVersionError`); you must depend on
  `rerun-sdk[datafusion]`. A clearer error pointing at the exact fix (it already names the extra —
  good) plus docs would help.
