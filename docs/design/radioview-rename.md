# RadioView rename plan

Status: approved name; implementation intentionally not started
Decision date: 2026-08-09

This document is the handoff for renaming the application to **RadioView**.
Do not implement it until the user explicitly authorizes the rename.

## 1. Canonical identity

- Product name: `RadioView`
- Exact spelling and capitalization: capital `R`, capital `V`, no space
- Short UI subtitle: `Multifrequency radio and context analysis`
- Python/npm distribution slug: `radioview`
- One-sentence description: `Interactive workbench for multifrequency solar radio imaging, spectroscopy, context visualization, and dynamics.`

The name is not an acronym. Do not expand it, append `Viewer`, or retain
`Feature/Radio Comparison Tool` as a secondary product name.

## 2. Scope and success criteria

This is a focused branding and package-metadata rename. It succeeds when:

1. The browser tab, application topbar, FastAPI metadata, root response,
   README, and package metadata consistently say `RadioView`.
2. The README describes the application as a general, role-based solar radio
   analysis workbench rather than a SAD/EOVSA comparison tool.
3. Existing manifests, saved sessions, API routes, imports, launch scripts,
   exports, and tests remain compatible.
4. The frontend builds and the complete backend test suite passes.
5. No unrelated visual, behavioral, or architectural change is included.

### Explicit compatibility boundary

Keep these identifiers unchanged in this rename:

- Python import package `sad_eovsa_tool`
- Class `SadEovsaSession` and existing internal method names
- Launcher target `sad_eovsa_tool.backend.app:app`
- Saved-state keys such as `sadTracks` and `eovsaSources`
- Generic exports `feature_tracks.csv` and `radio_sources.csv`
- Compatibility exports `sad_tracks.csv` and `eovsa_sources.csv`
- Manifest roles `context`, `radio`, and `spectrogram`
- Genuine instrument/data identifiers such as AIA, EOVSA, OVRO-LWA, SAD
  fields, loaders, source labels, and sample paths

Those names are compatibility contracts or scientifically meaningful source
identifiers, not the current application brand. A Python namespace migration
can be planned separately if it is ever needed.

## 3. Phase 0 — documentation discovery

The following source-of-truth surfaces were inspected before writing this
plan. Line numbers are the positions observed on 2026-08-09; use the named
symbols and strings if later edits move them.

| Source | Current contract |
| --- | --- |
| `README.md:1-3` | Public title and scope description |
| `frontend/src/App.tsx:3283-3289` | Visible topbar brand and subtitle |
| `frontend/index.html:6` | Browser-tab title |
| `sad_eovsa_tool/backend/app.py:208` | FastAPI/OpenAPI title |
| `sad_eovsa_tool/backend/app.py:1422-1424` | Root endpoint brand response |
| `pyproject.toml:5-9` | Python distribution name and description |
| `frontend/package.json:2` | Frontend package name |
| `frontend/package-lock.json:2,8` | Generated npm root-package metadata |
| `uv.lock:1707-1711` | Generated Python root-package metadata |
| `sad_eovsa_tool/__init__.py:1` | Package docstring |
| `sad_eovsa_tool/backend/__init__.py:1` | Backend package docstring |
| `sad_eovsa_tool/backend/app.py:1` | API module docstring |
| `sad_eovsa_tool/backend/data.py:1` | Data module docstring |
| `run_backend.sh:12` | Import-path constraint that must remain valid |
| `README.md:42-48` | Existing frontend/backend verification commands |

Allowed implementation mechanisms are deliberately small:

- Replace static UI and HTML text in their existing locations.
- Reuse the existing FastAPI `title` argument and root endpoint; do not add an
  endpoint or change the response schema.
- Change standard Python/npm package metadata and regenerate their lockfiles
  with the existing package managers.
- Update prose/docstrings without changing runtime behavior.

No new dependency, API, state migration, configuration layer, or branding
framework is needed.

## 4. Phase 1 — runtime and visible branding

### What to implement

1. Add a module-level `APP_NAME = "RadioView"` beside the FastAPI setup in
   `sad_eovsa_tool/backend/app.py`.
2. Use that constant in both existing brand surfaces:

   ```python
   APP_NAME = "RadioView"
   app = FastAPI(title=APP_NAME)
   ```

   and:

   ```python
   return {"app": APP_NAME, "frontend": "http://127.0.0.1:5174"}
   ```

3. Replace the brand block in `frontend/src/App.tsx` with:

   ```tsx
   <strong>RadioView</strong>
   <span>Multifrequency radio and context analysis</span>
   ```

4. Change `frontend/index.html` to `<title>RadioView</title>`.
5. Update the four package/module docstrings to describe `RadioView` without
   changing any symbol names.

### Test first

Add a small, data-independent regression test to the existing
`P3BackendTest` in `sad_eovsa_tool/backend/tests/test_p3_backend.py`, which
already owns a `TestClient` fixture:

```python
def test_root_reports_radioview_brand(self) -> None:
    response = self.client.get("/")
    self.assertEqual(response.status_code, 200)
    self.assertEqual(response.json()["app"], "RadioView")
    self.assertEqual(api.app.title, "RadioView")
```

Run this test before the implementation to prove it fails for the old name,
then rerun it after the edits.

### Verification checklist

- The new backend branding test passes.
- The browser tab reads `RadioView`.
- The topbar reads `RadioView` and its subtitle does not overflow at the
  existing desktop layout widths.
- `GET /` returns the same two keys as before, with only `app` changed to
  `RadioView`.
- OpenAPI remains available and shows `RadioView` as its title.

### Anti-pattern guards

- Do not rename routes, request/response fields, React state, or CSS classes.
- Do not change the `Sparkles` icon or redesign the topbar.
- Do not globally replace `SAD`, `EOVSA`, or `AIA`.
- Do not introduce a frontend branding abstraction for two static strings.

## 5. Phase 2 — README and package metadata

### What to implement

1. Change `README.md` heading to `# RadioView`.
2. Replace its opening paragraph with this copy:

   > RadioView is an interactive workbench for time-synchronized exploration
   > and analysis of multifrequency solar radio images, dynamic spectra, and
   > co-temporal context imagery. It supports layered display, temporal
   > differencing, WCS-aligned overlays, ROI selection, feature tracking, and
   > radio-source extraction. The bundled sample is the 2022-01-18 AIA 131 Å /
   > EOVSA M-flare event, but the application is instrument- and
   > event-independent through role-based `context`, `radio`, and optional
   > `spectrogram` sources.

3. In `pyproject.toml`, set:

   ```toml
   name = "radioview"
   description = "Interactive workbench for multifrequency solar radio imaging, spectroscopy, context visualization, and dynamics."
   ```

   Keep `[tool.setuptools.packages.find]` pointed at `sad_eovsa_tool*`; the
   distribution name and import package are allowed to differ.

4. Set the root `name` in `frontend/package.json` to `radioview`.
5. Regenerate, rather than hand-edit, the lockfiles:

   ```text
   uv lock
   npm install --package-lock-only --prefix frontend
   ```

6. Confirm the regenerated root entries in `uv.lock` and
   `frontend/package-lock.json` are `radioview` and that dependency versions
   did not change unexpectedly.

### Verification checklist

- `README.md` describes images, spectra, context, dynamics, tracking, and
  extraction without presenting SAD/EOVSA as the product boundary.
- `pyproject.toml`, `uv.lock`, `frontend/package.json`, and
  `frontend/package-lock.json` agree on `radioview`.
- `uv lock --check` succeeds when the installed uv version supports it.
- The editable install still exposes the `sad_eovsa_tool` Python import.
- Launch commands in the README and shell scripts remain unchanged and valid.

### Anti-pattern guards

- Do not rename the workspace directory or Python source directory.
- Do not hand-edit generated `frontend/dist`, `*.egg-info`, `.pytest_cache`,
  `.venv`, or log files.
- Do not upgrade dependencies while regenerating lockfiles.
- Do not rewrite historical design documents merely because they contain
  valid instrument-specific names.

## 6. Phase 3 — final verification

Run the focused stale-brand search, excluding this handoff document and
generated/cache directories:

```text
rg -n "Feature/Radio Comparison Tool|Feature / Radio Comparison|Solar imaging workbench|sad-eovsa-tool" \
  README.md pyproject.toml uv.lock frontend sad_eovsa_tool \
  -g '!frontend/node_modules/**' -g '!frontend/dist/**' \
  -g '!**/__pycache__/**' -g '!docs/design/radioview-rename.md'
```

Expected result: no matches. Do **not** use `sad_eovsa_tool` as a stale-brand
search term because that import namespace intentionally remains.

Then run:

```text
npm run build --prefix frontend
python -m pytest sad_eovsa_tool/backend/tests/
```

Launch the app with the existing `./run_app.sh` workflow and verify:

```text
curl http://127.0.0.1:8010/
curl http://127.0.0.1:8010/api/health
```

Final manual checks:

- Browser tab and topbar both show `RadioView`.
- Existing sample and manifest loading still work.
- A previously saved session still loads.
- Generic and legacy-compatible exports retain their existing filenames.
- No visible `Feature / Radio Comparison` brand remains.

## 7. Out of scope

- Renaming `sad_eovsa_tool` to a new Python import namespace
- Renaming `SadEovsaSession` or other internal classes/functions
- Removing legacy saved-state keys or export aliases
- Generalizing instrument-specific loaders or sample data
- Changing API routes or schemas
- Logo, icon, layout, or theme changes
- Publishing a package, reserving a domain, or trademark work

## 8. Coordination note for Claude/Codex

Treat this file as the shared implementation contract. Before starting, rerun
the Phase 0 searches because line numbers may have moved. Only one agent
should own the rename edits at a time. If concurrent work has changed one of
the listed files, preserve that work and adapt the smallest possible rename
diff around it.
