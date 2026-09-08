# Centralized layer rail

Status: implementation spec for the executing agent. Frontend-only — **no
backend changes**. Follows up the layered-panels work already implemented in
`frontend/src/App.tsx`; supersedes the in-panel layer UI it introduced.

## 0. Instructions for the implementing agent

- Everything in this spec happens in `frontend/` (`App.tsx`, `App.css`). Do
  not touch the backend, `frameScheduler.ts`, or `timeResolution.ts` except
  where explicitly named.
- The interaction pattern to follow is from the reference repo (read-only):
  `/path/to/projects/coronal_B_field_radio/repos/pyampp-web`
  — specifically the *field-line seeds* section of
  `frontend/src/components/LayerRail.svelte` (rows + editor-below-list +
  per-row ✕ delete). It is Svelte; copy the interaction design, not the code.
- Keep existing state names (`panelLayers`, `panelCompositions`,
  `selectedLayerId`, `LayerState`) and the update/remove/copy functions
  (`updatePanelLayer`, `removePanelLayer` App.tsx:1266, `copyPanelLayer`
  App.tsx:1279, `movePanelLayer`, `reorder`) — this is a UI reorganization,
  not a state-model rewrite, except for the deletion-rule change in §4.
- Verify with `npm run build` and the behavior checklist in §8. One commit
  per numbered section is fine; §1–§4 are the core and must land together.

## 1. What moves where (the core change)

**Problem** (user feedback on the current UI): each panel carries its own
layer stack UI (`PanelLayerControls`, App.tsx:3663-3715) with a toolbar
(Base select, + Image, + Overlay, Copy →) and rows whose inspector expands
*inline inside the row* (App.tsx:3694-3709). With several layers this
crowds the image panels, the inline inspector overflows the panel width
(controls get clipped at the right edge), and every layer repeats the same
knobs. Meanwhile the left rail still shows the pre-layer legacy knob
cluster (App.tsx:2602-2675: DifferenceControls, X/Y offsets, "All Bands on
Context", radio colormap, Open/Filled, Current/Global + refresh, level
units, level, opacity) — duplicating what layer inspectors already control.

**Target**:

1. A single **Layers** section in the left rail manages both panels' stacks.
2. Panels lose all layer chrome — `PanelLayerControls` is removed from the
   panel headers entirely; panels show only title, timestamp, zoom/FIT
   controls, and the canvas.
3. The legacy contour/difference knob cluster in the rail is deleted; the
   layer inspector is the single home for those controls (§5).
4. One **inspector card** at the bottom of the Layers section shows the
   selected layer's configuration (§3). No inline row expansion anywhere.

## 2. The Layers rail section

Placement: new `tool-panel` section directly below the Data section, above
Time. Structure:

```
▾ Layers                                   (collapsible section header)
  Left panel                 [+ Image] [+ Overlay]
    ◉ ⣿ AIA 131 Å           [base]  Δ0.0s     ✕
    ◉ ⣿ Radio contours              Δ0.4s     ✕
  Right panel                [+ Image] [+ Overlay]
    ◉ ⣿ EOVSA 1.10–1.73 GHz [base]  Δ857s     ✕
    ○ ⣿ AIA 131 Å                   unavail.  ✕
  ┌─ Inspector: Left · Radio contours ──────────┐
  │  … per-kind controls (§3) …                 │
  │  [Set as base] [Copy to right panel] [Delete]│
  └──────────────────────────────────────────────┘
```

Row anatomy (compact, one line, pyampp-style):

- **Eye** toggle (`◉`/`○`) — visibility; reuse current glyphs (App.tsx:3678).
- **Drag handle** (`⋮⋮`) — keep the existing HTML5 drag reorder
  (App.tsx:3676-3677) *within the same panel group only*; keep ↑/↓ reorder
  as inspector buttons for accessibility (drop the per-row ↑↓ buttons).
- **Name button** — click selects the layer and shows the inspector card;
  clicking the already-selected row **deselects** and hides the card
  (pyampp toggle semantics, `LayerRail.svelte` `selectSeed`). Name text is
  the layer label; a small `base` badge marks the base image layer
  (replaces the toolbar Base `<select>`; see §4).
- **Status** — the existing resolved-time/Δt/Unavailable/Pending text
  (App.tsx:3680-3686), abbreviated to fit one line (`Δ0.4s`,
  `unavailable`); the full resolved UTC string moves into the inspector.
- **✕ delete** — always visible, right-aligned, immediate (no confirmation
  dialog — pyampp behavior). Enablement per §4.

What does **not** appear in rows anymore: the per-row source `<select>`
(App.tsx:3687-3689) and opacity slider (App.tsx:3690) move into the
inspector. Rows stay scannable.

Group toolbar: `+ Image` and `+ Overlay` per panel group (same handlers as
today's `onAdd`, App.tsx:3670-3671). Adding a layer **auto-selects it**
(pyampp behavior) so the inspector opens on the new layer. `Copy →` leaves
the toolbar and becomes an inspector button ("Copy to left/right panel",
same `copyPanelLayer` handler).

Density: each group's list gets `max-height: ~9 rows` with inner
`overflow-y: auto` so many layers never stretch the rail. Empty group shows
muted text: `no layers — add an image layer` (pyampp empty-state pattern).

Selection state: keep `selectedLayerId` but make it global-single —
`{ slot: PanelSlotId; id: string } | null` — so only one inspector exists
across both groups (today it is per-slot, allowing two open inline
inspectors at once).

## 3. The inspector card

One card, rendered at the bottom of the Layers section, only when a layer
is selected. Header: `<Left|Right> · <layer label>`, plus the full resolved
timestamp line (`2025-03-28 15:40:20.072 UTC · Δ0.0s` / `Unavailable`).

Controls (reuse the exact existing bindings from the inline inspector,
App.tsx:3696-3707, laid out as a labeled 2-column grid instead of a
squeezed row):

- All kinds: **Source** (the select moved from the row), **Opacity**
  (slider moved from the row), Operation, Reference, Sample policy,
  Tolerance [s]; Lag [s] when reference = previous; Mean start/end when
  reference = mean.
- Image layers: vmin, vmax, Map, Scale, Freq (radio sources only — keep the
  current conditional).
- Contour layers additionally: Contour map (with the gradient swatch from
  the legacy rail control, App.tsx:2617), Level mode, Current/Global
  reference **plus the global-peak refresh button** (move it from the
  legacy cluster, App.tsx:2650-2652 — it must not be lost), Level value,
  Contour opacity, Open/Filled as a two-button toggle (keep the legacy
  two-button style, App.tsx:2628-2635, rather than the current checkbox).
- Action row at the card bottom: `Set as base` (image layers only; replaces
  the Base dropdown), `Copy to <other> panel`, `Move up` / `Move down`,
  `Delete` (red/danger style; same enablement as the row ✕).

Numeric inputs keep `CommitInput` (commit-on-blur/Enter) — no
per-keystroke refetches.

## 4. Deletion rules (fix "layers cannot be deleted")

Today `canRemoveLayer` (App.tsx:3732-3738) disables ✕ for the base layer
when no other image layer exists, and for the last layer — silently. With
one base + one contour overlay (the default composition), the base's ✕ is
disabled with no explanation, which reads as "delete doesn't work".

New rules:

1. **Every layer is deletable, always.** No disabled ✕.
2. Deleting the base image layer promotes the next image layer in that
   panel (in stack order) to base (`compositionFor` already resolves a
   fallback base — extend `removePanelLayer` App.tsx:1266-1277 to drop its
   base/last-layer guards).
3. Deleting the last image layer while contour overlays remain is allowed;
   the contour layers mark unavailable (the existing
   `isLayerUnavailable`/status path handles rendering).
4. Deleting the last layer of a panel is allowed; the panel canvas shows an
   empty state (`No layers — add one in the Layers panel`) and the panel
   group shows the empty-state text of §2.
5. Deleting the selected layer moves selection to the next remaining layer
   in that group, or clears selection if none (pyampp `removeSeed`
   behavior, already approximated at App.tsx:1276 — keep).

## 5. Retire the legacy rail knobs

In the Data section's `source-knobs` block:

- **Delete the whole radio contour cluster** (App.tsx:2611-2674): "All
  Bands on Context" button, Radio colormap, Open/Filled, Current/Global +
  refresh, Level units, Level, Opacity. All of it now lives in the contour
  layer's inspector (§3). The `showEovsaContours` state and the global
  contour states (`contourCmap`, `contourFilled`, `contourLevel*`,
  `contourOpacity`) become dead once nothing reads them — remove them and
  their `SavedUiState` writes, but **keep reading them on load** as
  seeds for a migrated contour layer (legacy sessions with
  `showEovsaContours: true` must still open with a visible contour layer
  configured from those values; this load path likely already exists from
  the layer migration — verify, don't duplicate).
- **Remove per-source `DisplayControls` + `DifferenceControls`** from
  source-knobs (they duplicate the layer inspector). The Data section's
  per-source area keeps only: role select, and the X/Y offset inputs
  **moved out** — offsets are cross-layer alignment, so relocate the two
  offset inputs (App.tsx:2607-2610) into the ROI/Alignment tool panel
  (which already has offset inputs — dedupe to a single pair there; keep
  the `refreshBothRoiProjections` commit behavior).
- `sourceDifferences` remains as the *default seed* for newly added layers
  of that source (current add-layer behavior) — do not remove the state,
  only its rail UI.

## 6. Collapsible rail sections

Net-new (the reference app lacks this; the user wants it):

- Every rail `tool-panel` section header (Data, Layers, Time, ROI/
  Alignment, Feature Tracking, Exports) becomes a disclosure toggle:
  chevron (`▾`/`▸`) + heading, click toggles the section body.
  `aria-expanded` on the button, body hidden via CSS class (no unmount —
  keep effects alive).
- Persist open/closed as `ui.collapsedSections: string[]` in the saved
  session state (default: all open). Absent field → all open (legacy
  compat).

## 7. Saved state

- `ui.panels`/`ui.layers` serialization is unchanged.
- Add: `ui.collapsedSections` (§6). Selection is not persisted.
- Remove: the global contour fields from *writing* (§5) while still
  *reading* them for legacy migration. Keep writing all other existing
  fields.

## 8. Definition of done

- `npm run build` passes; no new dependencies.
- Panels show no layer chrome; the workspace area contains only
  spectrogram + two clean image panels.
- With 6+ layers on one panel, the rail group scrolls internally; rows stay
  one line each; nothing overflows horizontally.
- Behavior checklist:
  - Click layer name → inspector card appears at section bottom; click the
    same name again → card closes.
  - Only one inspector can be open across both panels.
  - `+ Image`/`+ Overlay` adds and auto-selects; inspector opens.
  - Every ✕ deletes immediately; deleting a base promotes the next image
    layer (badge moves); deleting the last layer yields both empty states
    (§4.4); selection follows §4.5.
  - "Set as base" swaps the base badge and panel rendering accordingly.
  - Contour layer inspector contains colormap+swatch, Open/Filled buttons,
    Current/Global + working global-peak refresh, level mode/value,
    contour opacity — and the panel updates on each change.
  - Legacy saved JSON (pre-layer format with `showEovsaContours: true`)
    still opens with a visible, correctly-configured contour layer.
  - All six rail sections collapse/expand; state survives export → reload
    via `ui.collapsedSections`.
  - X/Y offsets exist exactly once, in ROI/Alignment, and still trigger
    ROI re-projection on commit.
- Screenshot the rail with the Layers section open (3+ layers, one
  selected) and include it in your report.

## 9. Out of scope

No backend changes; no changes to `frameScheduler.ts` request behavior; no
master-clock/Time-section changes; no spectrogram-panel changes; no visual
redesign beyond what this spec names (design tokens etc. remain a later
polish phase).
