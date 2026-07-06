# Releasing `rerun-lerobot`

This is the release checklist for maintainers. Replace `0.NEW.VERSION` with the
version you are releasing (e.g. `0.2.0`).

Do everything from a clean checkout of `main` with a working `uv` install.

## Update to the latest Rerun and LeRobot

`rerun-lerobot` tracks two moving targets:

- **`rerun-sdk`** — the OSS server API (`rr.server.Server`) needs `>= 0.27`, and
  the catalog client needs a matching `datafusion` (pulled in via the
  `rerun-sdk[datafusion]` extra — do **not** depend on `datafusion` directly).
- **`lerobot`** — the conversion relies on its private dataset internals, so it
  is pinned `>=X.Y,<X.(Y+1)` (e.g. `>=0.6.0,<0.7`). Re-verify against those
  internals before widening the bound. Since LeRobot 0.6 only constrains
  `rerun-sdk` under its optional `viz` extra (which we don't install), there is
  no longer a `rerun-sdk` conflict to override.

To upgrade:

1. Bump the bounds in `pyproject.toml` `dependencies`:
   - `lerobot[dataset]>=X.Y.Z,<X.(Y+1)` — bump within the pinned major, or widen
     the cap only after re-verifying the private-API usage in `converter.py`.
   - `rerun-sdk[datafusion]>=A.B` if the minimum Rerun changed.
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

## Make sure CI is green

Push your changes and confirm the `Python` workflow passes on the release commit
before tagging or publishing — a green local run is not enough.

```bash
git push
gh run watch "$(gh run list --workflow python.yml --limit 1 --json databaseId --jq '.[0].databaseId')" --exit-status
```

Do not proceed to publish/tag until CI reports `success`.

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

Publishing is automated. The `Release` workflow (`.github/workflows/release.yml`)
runs when a version tag is pushed, builds the package with `uv build`, and
uploads it to PyPI via **Trusted Publishing (OpenID Connect)** — no API token is
stored anywhere. GitHub mints a short-lived OIDC token that PyPI verifies against
the trusted publisher configured for this project.

One-time setup (already done; only needed if the repo or project is recreated):

- **PyPI** — on the project's *Publishing* page add a trusted publisher:
  owner `rerun-io`, repo `rerun-lerobot`, workflow `release.yml`, environment
  `release`.
- **GitHub** — create an environment named `release`
  (Settings → Environments). Optionally restrict it to release tags and require
  reviewers, so only vetted tags publish.

To release, just tag and push (see below). Watch the run:

```bash
gh run watch "$(gh run list --workflow release.yml --limit 1 --json databaseId --jq '.[0].databaseId')" --exit-status
```

The workflow refuses to publish if the tag does not match `__version__`.

Once published, verify the release installs cleanly from PyPI into a throwaway
env (requires Python >= 3.12; no dependency override needed):

```bash
uv venv /tmp/verify-rl --python 3.12
uv pip install --python /tmp/verify-rl "rerun-lerobot==0.NEW.VERSION"
/tmp/verify-rl/bin/rerun-lerobot --help
```

## Tag the release

Pushing the tag triggers the `Release` workflow, which publishes to PyPI:

```bash
git push origin main
git tag -a 0.NEW.VERSION -m "Release 0.NEW.VERSION - <one-line summary>"
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
  --generate-notes
```

`--generate-notes` fills the changelog from merged PRs; edit it to add a short
summary and call out the Rerun / LeRobot versions this release was built against.
The build artifacts live on PyPI (the source of truth), so nothing needs to be
attached here.
