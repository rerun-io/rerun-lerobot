# Releasing `rerun-lerobot`

This is the release checklist for maintainers. Replace `0.NEW.VERSION` with the
version you are releasing (e.g. `0.2.0`).

Do everything from a clean checkout of `main` with a working `uv` install.

## Update to the latest Rerun and LeRobot

`rerun-lerobot` tracks two moving targets:

- **`rerun-sdk`** — the OSS server API (`rr.server.Server`) needs `>= 0.27`, and
  the catalog client needs a matching `datafusion` (pulled in via the
  `rerun-sdk[datafusion]` extra — do **not** depend on `datafusion` directly).
- **`lerobot`** — pins `rerun-sdk < 0.27`, so its pin is overridden in
  `[tool.uv] override-dependencies` in `pyproject.toml`.

To upgrade:

1. Bump the lower bounds in `pyproject.toml`:
   - `dependencies`: `lerobot>=X.Y.Z` and, if the minimum Rerun changed,
     `rerun-sdk[datafusion]>=A.B`.
   - `[tool.uv] override-dependencies`: keep `rerun-sdk[datafusion]>=A.B` in sync.
2. Re-resolve and install:
   ```bash
   uv lock --upgrade
   uv sync --dev
   ```
3. Confirm nothing broke:
   ```bash
   uv run ruff format --check
   uv run ruff check
   uv run mypy
   uv run pytest          # runs the end-to-end conversion test
   ```
4. Commit the updated `pyproject.toml` and `uv.lock`.

If a new `rerun-sdk` moves `datafusion` to a new major version, `uv run pytest`
will fail with `RerunIncompatibleDependencyVersionError` — that means the
`rerun-sdk[datafusion]` extra is doing its job; just re-run `uv lock --upgrade`.

## Bump the version number

The version lives in a single place — `rerun_lerobot/__init__.py`
(`__version__`); Hatchling reads it dynamically.

1. Edit `rerun_lerobot/__init__.py` and set `__version__ = "0.NEW.VERSION"`.
2. Sanity check:
   ```bash
   uv run python -c "import rerun_lerobot; print(rerun_lerobot.__version__)"
   ```
3. Commit:
   ```bash
   git commit -am "Release 0.NEW.VERSION - <one-line summary>"
   ```

## Publish to PyPI

Build and publish from the clean, committed tree:

```bash
rm -rf dist
uv build                 # produces dist/*.whl and dist/*.tar.gz
uv publish               # uploads to PyPI
```

`uv publish` needs a PyPI token. Either export it first:

```bash
export UV_PUBLISH_TOKEN=pypi-...
```

or pass `--token pypi-...`. To test the upload first, publish to TestPyPI with
`uv publish --publish-url https://test.pypi.org/legacy/`.

Verify the release installs cleanly (note the `rerun-sdk` override, explained in
the [README](README.md)):

```bash
uv run --with "rerun-lerobot==0.NEW.VERSION" --with "rerun-sdk>=0.27" \
  --no-project -- rerun-lerobot --help
```

## Tag the release

```bash
git tag -a 0.NEW.VERSION -m "Release 0.NEW.VERSION - <one-line summary>"
git push origin main
git push origin 0.NEW.VERSION
```

## Make a GitHub release

Create the release from the tag, either in the UI:

<https://github.com/rerun-io/rerun-lerobot/releases/new>

or with the `gh` CLI:

```bash
gh release create 0.NEW.VERSION \
  --repo rerun-io/rerun-lerobot \
  --title "0.NEW.VERSION" \
  --generate-notes \
  dist/*.whl dist/*.tar.gz
```

`--generate-notes` fills the changelog from merged PRs; edit it to add a short
summary and call out the Rerun / LeRobot versions this release was built against.
Attaching the built wheel and sdist is optional (PyPI is the source of truth) but
convenient.
