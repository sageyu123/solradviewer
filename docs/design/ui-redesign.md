# UI redesign: workstation layout

Status: implementation spec for the executing agent. Frontend-only — no
backend changes. This is the master spec; it incorporates
[`centralized-layer-rail.md`](centralized-layer-rail.md) as **Part A**
(normative, read it first) and adds Parts B–E around it.

Design intent, in one paragraph: the data views own the screen and every
control lives in exactly one place. The app currently carries three
generations of controls (per-source knobs, the contour cluster, in-panel
layer stacks) with the same setting reachable from multiple places. The
redesign consolidates to: one collapsible rail with a single inspector
(Part A), the dynamic spectrum as the master timeline (Part B), image
panels as pure canvases (Part C), one status affordance in the topbar
(Part D), and the visual language of the reference app
`ovrolwa-rfr-corr-app` (Part E).

## 0. Instructions for the implementing agent

- All work in `frontend/` (`App.tsx`, `App.css`). Backend, `frameScheduler.ts`,
  `timeResolution.ts` untouched except where a part explicitly says.
- Reference repo (read-only, for interaction + visual patterns):
  `/path/to/projects/ovrolwa-rfr-corr-app`
  — cited below as `ref:`. It is also React + vanilla CSS; its patterns
  transfer directly.
- **Implementation order: A → B → C → D → E.** One part per commit minimum.
  E (theming) is last so nothing gets styled twice. Stop and report after A+B
  with screenshots, then continue.
- Verify each part with `npm run build` plus its DoD checklist. Screenshot
  the full app after B and after E and include both in your report.
- No new dependencies. When ambiguous, choose the smallest compliant change
  and note the ambiguity in your report.

## Part A — Centralized layer rail

Normative spec: [`centralized-layer-rail.md`](centralized-layer-rail.md)
(sections 1–9, including deletion rules, legacy-knob removal, collapsible
sections, and its DoD). Implement it as written. One amendment: its §6
collapsible-section behavior should use the section-header style adopted in
Part E (chevron + uppercase small heading with trailing rule,
`ref: frontend/src/app.css:43-48`).

## Part B — Dynamic spectrum as master timeline

The spectrogram panel already renders a time-frequency image with a cursor.
Make it the primary time-navigation surface, adopting the reference app's
`TimeSeriesChart` interactions (`ref: frontend/src/components/TimeSeriesChart.tsx`,
pointer wiring at :865-1002):

1. **Scrub**: dragging on (or near) the cursor line moves the master cursor
   continuously; a plain click jumps to that time (the existing
   `selectSpectrogramTime` path, App.tsx generalized to the master clock).
   Scrubbing feeds the same debounced commit path as the slider (Part A/
   frameScheduler behavior unchanged).
2. **Wheel zoom on the time axis**: wheel over the spectrogram zooms the
   time window around the pointer; drag on empty area pans the window.
   Keep the existing box-zoom and FIT buttons; they now share one
   view-window state with wheel/pan.
3. **Transport header**: the spectrogram panel header gains, left-aligned:
   play/pause, an fps `<select>` (0.5 / 1 / 2 / 5 / 10 / 20 fps — replaces
   the fixed 180 ms interval; `ref: App.tsx:505-522, 1375-1382` for the
   `setTimeout`-chain pattern), and prev/next frame-step buttons. Playback
   advances the master cursor and is load-chained (next step waits for the
   current frame), which Part A's scheduler already enables.
4. **Cursor rendering**: a full-height cursor line with a small grab handle,
   plus the current UTC timestamp near the header (single source of truth
   for "now").

**The Time rail section is kept** (user decision): slider + timestamp
readout + frame-by-frame step buttons + start/end range inputs + master
clock source select + sampling defaults. It and the spectrogram scrub are
two views of the same master-cursor state; moving either updates both. The
transport buttons appear in the spectrogram header only (not duplicated in
the rail beyond the existing step buttons).

DoD (B): scrubbing on the spectrogram pans frames with the same
responsiveness as the slider; wheel-zoom + pan + box-zoom + FIT compose
without fighting; playback runs at the selected fps without skipping on
uncached spans; slider and spectrogram cursor never disagree.

## Part C — Panels as pure canvases

After Part A removed the layer stacks, finish the job:

- Panel header contains exactly: layer-derived title (base layer label, plus
  `+ contours` suffix when contour overlays are visible), resolved timestamp
  with signed Δt, and the zoom controls (−, FIT, +). Nothing else.
- Remove the bottom status bar row from the workspace grid
  (App.tsx:1884-1887 region; its message/progress duties move to Part D).
  The reclaimed 30 px goes to the panels.
- Empty state (no layers in a panel, per Part A §4.4): centered muted text
  `No layers — add one in the Layers panel`.
- Keep: shared `solarView` pan/zoom sync between the two panels, ROI lasso,
  tracks/sources drawing, colorbar (now per contour overlay, from Part A).

DoD (C): with the rail collapsed, the workspace is spectrogram + two clean
canvases and nothing else; no residual layer or status chrome.

## Part D — Centralized status (topbar)

Adopt the reference app's session-status pattern
(`ref: App.tsx:1193-1266`):

- Topbar right side: a status dot (idle / busy / error color via tokens)
  + one-line status text; during long operations (peak-cache refresh,
  extraction) an inline determinate progress bar with `completed/total`
  when counts exist, indeterminate otherwise.
- Errors: dismissible toast-style banner overlaying the workspace top edge
  (`ref: App.tsx:1790-1795`), auto-dismiss after ~8 s, never a blocking
  alert. Replace current inline error text paths.
- The existing `loading` boolean and status message state map onto this;
  no new state model needed — this is a presentation change.

DoD (D): every message that used to appear in the bottom bar appears in the
topbar status or a toast; long operations show progress; errors are
dismissible and don't shift layout.

## Part E — Visual refresh (design tokens + theme)

Goal: the three apps (this one, `ovrolwa-rfr-corr-app`, `pyampp-web`) read
as one family. Adopt the reference app's token system wholesale
(`ref: frontend/src/styles.css:1-17`):

1. **Tokens**: define in `:root` (App.css): `--topbar-height`, `--rail-width`,
   `--panel: #202328`, `--panel-raised: #292d32`, `--border: #3b4046`,
   `--muted: #979ea6`, `--text: #e7e9eb`, accent `--cyan: #56c7d9`,
   secondary accent `--magenta: #ee65be`. Replace every hardcoded color in
   `App.css` with a token. Keep the current dark theme as the only theme.
2. **Section headers**: uppercase, small, muted, trailing horizontal rule
   (`ref: app.css:43-48`), doubling as the Part A disclosure toggles.
3. **Controls**: one consistent style for buttons/selects/inputs — quiet by
   default (panel-raised background, token border), accent (`--cyan`) only
   for the single active/primary state per cluster (e.g. active mode
   button, play while playing). The current amber "active" buttons are
   replaced by this. At most one accent color per control cluster.
4. **Density**: rail controls at 12–13 px labels, compact row heights
   matching the layer rows of Part A; panels' headers same height as the
   reference app's panel headers.
5. **Focus + motion**: visible `:focus-visible` rings on all interactive
   elements; `@media (prefers-reduced-motion: reduce)` disables the few
   transitions (`ref: styles.css:2412`).
6. **Keyboard map** (net-new, guarded against focused inputs/dialogs,
   `ref: App.tsx:524-556`): Left/Right = frame step, Shift+Left/Right = ×10,
   Space = play/pause, Home/End = range start/end. Document in a topbar
   tooltip (`?` icon).

DoD (E): zero hardcoded hex colors left in `App.css` outside the `:root`
block; side-by-side screenshots of this app and the reference app show
consistent chrome (topbar, rail sections, buttons); keyboard map works and
never fires while typing in an input.

## Saved state (whole redesign)

- From Part A: `ui.collapsedSections`.
- New: `ui.playbackFps` (default 5). Timeline zoom window is *not*
  persisted.
- Everything else unchanged; legacy sessions must load exactly as specified
  in Part A's spec §5/§7.

## Out of scope

Backend; request scheduling (`frameScheduler.ts`); master-clock semantics
(`timeResolution.ts`); ROI/tracking/extraction logic; light theme; guided
tour; mobile layouts. These stay as they are.
