#!/usr/bin/env node
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const app = fs.readFileSync(path.join(root, "src/App.tsx"), "utf8");
const css = fs.readFileSync(path.join(root, "src/App.css"), "utf8");
for (const token of ["topbar-status", "topbar-message", "status-dot", "toast-error", "beforeunload", "confirmDatasetReplace", "empty-workspace", "ArrowLeft", "ArrowRight", "baselineSignatureRef", "role=\"progressbar\"", "aria-valuetext=\"Working\""]) {
  assert.ok(app.includes(token), `P6 app token missing: ${token}`);
}
for (const token of ["prefers-reduced-motion", "--panel", "--border", ".topbar-status", ".toast-error"]) {
  assert.ok(css.includes(token), `P6 style token missing: ${token}`);
}
const ignored = (tag) => ["INPUT", "SELECT", "TEXTAREA", "BUTTON"].includes(tag);
assert.equal(ignored("INPUT"), true);
assert.equal(ignored("CANVAS"), false);
const dirtyBlock = app.slice(app.indexOf("const dirtySignature"), app.indexOf("const contextSource"));
for (const transient of ["panelLayers", "panelCompositions", "masterCursorMjd", "masterStartMjd", "masterEndMjd", "freqIndex"]) {
  assert.equal(dirtyBlock.includes(transient), false, `transient state incorrectly dirties session: ${transient}`);
}
for (const artifact of ["roiWorld", "roiAia", "roiEovsa", "sadTracks", "eovsaSources"]) {
  assert.equal(dirtyBlock.includes(artifact), true, `analysis artifact missing from dirty state: ${artifact}`);
}
assert.ok(app.includes("baselinePendingRef.current = false;\n      reportError(error);"), "load failure does not clear pending baseline");
assert.ok(app.includes("target.closest(\"input, select, textarea, button, a, [contenteditable='true'], dialog, [role='dialog']\")"), "Space capture lacks guarded targets");
assert.ok(app.includes("target.closest(\".spectrogram-canvas\")"), "spectrogram Space branch missing");
assert.ok(app.includes("Space play/pause outside images · Space pans over images"), "contextual keyboard tooltip missing");
assert.ok(app.indexOf("function keyUp(event: KeyboardEvent)") < app.indexOf("pointerOverImageRef.current || spaceDown"), "Space keyup guard missing");
console.log("P6 accessibility/status/guard probe passed.");
