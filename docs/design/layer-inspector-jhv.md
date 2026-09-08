# Layer inspector redesign — JHelioviewer-style rows

Status: implementation spec — mostly frontend (App.tsx + App.css) plus two
small backend additions (frame-stats headers §Row-grammar-4; instrument
colormap registration §Instrument-colormaps). Approved by the user against
a sketch; the reference look is JHelioviewer's layer
adjustment panel: one aligned control column — label left, control middle,
value right — radio rows for modes, and a "More adjustments" disclosure.

## Row grammar

Grid: `[right-aligned label | control | value box]`, label column ~88 px,
value column ~60 px. Three row types:

1. **Slider row** — for BOUNDED quantities. Slider in the middle, and an
   **editable numeric text box** (CommitInput, commit on Enter/blur) as the
   value readout on the right. Slider and box are two-way bound; the box
   accepts values outside the slider's range (slider pins to its end, the
   typed value wins). Sliders: Opacity (0–1, step .05), Coronal
   enhancement (0–3, step .1), Contour level % (1–99), Contour opacity
   (0–1), Temporal σ short (1–120 s), Temporal σ long (10–600 s).
2. **Radio row** — for 2–3-way modes, JHV-style inline radios:
   Operation (Original / Subtract / Ratio), Scale (Linear / Log),
   Temporal smoothing (None / Low-pass / Band-pass),
   Peak reference (Current / Global ⟳), Level mode (% peak / Kelvin —
   indented, shown only when Global), Fill (Open / Filled).
3. **Field row** — text/select for unbounded or discrete values: Source,
   Reference (select, inline on the Operation row with lag), Kelvin
   level, Lag [s], Band (radio image layers only, with GHz readout
   `Band 9 · 1.42 GHz`).
4. **Range row (vmin/vmax)** — a JHV-Levels-style DUAL-KNOB slider with an
   editable text box at each end. Track bounds are data-driven, not
   hardcoded:
   - Backend addition: frame responses gain headers `X-Data-Min`,
     `X-Data-Max`, `X-Data-P1`, `X-Data-P99` (robust stats of the rendered
     array BEFORE normalization, after difference/radial/temporal
     processing; ~ms cost; add to `Access-Control-Expose-Headers`).
   - Track bounds = `[min(p1, vmin), max(p99, vmax)]` of the most recent
     frame, recomputed ONLY when operation/reference/temporal/radialGamma
     change or on fit-press — never per playback frame (no track jitter).
   - Knobs = vmin/vmax; the text boxes remain authoritative and may exceed
     the track (knob pins to the end).
   - A small `fit` icon button beside the row sets vmin/vmax to the
     current frame's p1/p99 (same semantics as the light-curve card's
     Apply button).

## Layout order

**Image layers**: Source · Band (radio only) → Operation row (+ Reference
select + Lag inline) → Opacity → vmin → vmax → Color (map select + scale
radios inline) → Coronal enhancement → Temporal smoothing radio row → σ
short slider (visible when mode ≠ none) → More adjustments ▾ → actions.

**Contour layers**: Source (`(all bands)` note, NO Freq/Band field) →
Operation row → Opacity → Contour map (swatch + select) → Peak reference
radio row (Current / Global + refresh button) → indented Level-mode radio
row (only when Global; Current forces % peak) → Level slider (%) OR Kelvin
text field per mode → Contour opacity → Fill radio row → More
adjustments ▾ → actions.

**More adjustments** (collapsed by default, per layer): Sample policy,
Tolerance [s], Mean start/end (when reference = mean), Temporal σ long
(band-pass only). Disclosure state is transient (not persisted).

**Actions footer** unchanged: Set as base (image only) · Copy → · ↑ ↓ ·
Delete (danger, right-aligned).

## Instrument colormaps

- Backend: `import sunpy.visualization.colormaps` (sunpy is already a
  dependency; the import registers `sdoaia94/131/171/193/211/304/335` with
  matplotlib). Add `CMAP_ALIASES` entries `aia94..aia335 -> sdoaiaXXX`.
- Frontend: the image-layer Map select gains an "Instrument" optgroup with
  `AIA 94 Å … AIA 335 Å` entries (values `aia94`…). Available for any
  image layer; most meaningful for `operation: none` AIA layers.
- Do NOT change any existing default colormap; purely additive options.

## Behavior rules

- Every slider's readout is an editable CommitInput — no display-only
  numbers anywhere in the inspector.
- Level-mode nesting: switching Peak reference to Current forces
  levelMode = percent (Kelvin option hidden); switching to Global restores
  the last-used mode.
- Freq/Band field only on radio-source image layers; GHz readout derives
  from `meta.eovsa.freqGhz[freqIndex]`.
- No state-model changes: this is a re-rendering of the existing
  LayerState fields; saved-state format unchanged.
- Radio rows use the existing quiet-button style with the single cyan
  active state (Part E tokens); alignment via one shared CSS grid class.

## Compact mode revision (user feedback on v1)

The v1 rows waste label-gutter width at rail widths ~312 px: long labels
("Coronal enhancement", "Temporal smoothing") wrap, the Operation row
collides. Revision: **symbols + tooltips over text**, everywhere.

1. **Label column shrinks to ≤56 px.** Every row label becomes either a
   short word (≤7 chars) or a lucide icon (already imported in App.tsx)
   at 14–15 px with `title` tooltip carrying the full name:
   - Source → `Src`; Operation → `Op`; Opacity → Droplet icon
     (tooltip "Layer opacity"); vmin/vmax dual row → `Range`;
     Color/scale → Palette icon; Coronal enhancement → Sun icon
     (tooltip "Coronal enhancement — radial filter, γ 0–3");
     Temporal smoothing → Activity/Waves icon (tooltip incl. the
     attenuation caveat, replacing the inline caption text);
     σ short → `σ [s]`; σ long (More adjustments) → `σₗ [s]`;
     Contour map → same Palette icon; Peak reference → `Peak`;
     Level mode → `Units`; Level → `Level`; Contour opacity → Droplet;
     Fill → `Fill`; Lag → `Lag [s]`; Tolerance → `Tol [s]`.
2. **Radio-row options compress**: Original/Subtract/Ratio →
   `Orig · Sub · Ratio`; Low-pass/Band-pass → `LP · BP` (tooltips spell
   them out); % peak/Kelvin → `%` · `K`; Current/Global → `Cur` · `Glob`;
   Open/Filled → `Open` · `Fill`. Linear/Log stay as-is (already short).
3. **Action row becomes an icon strip** (5 icon buttons, existing
   icon-button style, `title` + `aria-label` on each): Set-as-base →
   Anchor (or Star), Copy-to-other-panel → Copy with direction arrow,
   Move up → ArrowUp, Move down → ArrowDown, Delete → Trash2 in danger
   color. No text buttons remain in the inspector footer.
4. **No inline caption paragraphs** in the inspector (the temporal
   attenuation note moves into the icon tooltip).
5. Long text elsewhere in the rail gets the same treatment where trivial
   (e.g. group add buttons stay, but `Copy to right panel` phrasing is
   gone everywhere in favor of the icon).

## Layer renaming

- `LayerState.label` becomes user-editable: a pencil icon in the
  inspector header (and double-click on the rail row name) swaps the name
  for an inline text input (Enter commits, Esc cancels).
- Add `labelEdited: boolean` to LayerState (persisted): once true, the
  label is never auto-regenerated on source change; false keeps today's
  auto-derived labels.
- Copy naming: replace the compounding `" copy"` suffix with numbered
  suffixes — `<label> (2)`, `(3)`, … computed against existing labels in
  both panels; copies of user-renamed layers keep the custom stem.
- Panel headers already derive from the base layer label, so renames flow
  to panel titles automatically — verify, don't reimplement.
- Saved-state: `label`/`labelEdited` round-trip (label already persists;
  add the flag; absent flag on old exports → false).

## Definition of done

- Both layer kinds render per the layouts above; no field appears twice;
  Freq absent for contours; Kelvin unreachable when Peak reference =
  Current.
- Typing 5 into the Coronal enhancement box pins the slider at 3 and
  applies 5 (backend accepts it); typing 0 restores identity.
- Slider drags and box commits produce identical state updates (single
  code path).
- `npm run build` passes; no new dependencies; behavior of every control
  identical to before (this is layout + the nesting rule, not semantics) —
  except the two bugfixes that may land first (gamma autofill guard,
  percent+global contours), which must not be reverted.
