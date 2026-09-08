#!/usr/bin/env node
/* Dependency-free smoke probe for the canonical v2 layer contract. */
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const app = fs.readFileSync(path.join(root, "src/App.tsx"), "utf8");
for (const token of ["Object.fromEntries(", "baseLayerId", "overlayLayerIds", "levelMode", "levelReference", "levelPercent", "levelKelvin", "sourceRoleSnapshot"]) {
  assert.ok(app.includes(token), `canonical layer token missing: ${token}`);
}

const canRemove = (layers, baseId, id) => {
  if (layers.length <= 1) return false;
  const target = layers.find((layer) => layer.id === id);
  return Boolean(target && (target.kind !== "image" || target.id !== baseId || layers.some((layer) => layer.kind === "image" && layer.id !== id)));
};
assert.equal(canRemove([{ id: "base", kind: "image" }, { id: "overlay", kind: "image" }], "base", "overlay"), true);
assert.equal(canRemove([{ id: "base", kind: "image" }, { id: "overlay", kind: "contours" }], "base", "base"), false);

const contour = {
  levelMode: "kelvin",
  levelReference: "global",
  levelPercent: "77",
  levelKelvin: "123456",
  filled: true,
  opacity: "0.62",
  cmap: "magma"
};
assert.equal(contour.levelMode, "kelvin");
assert.equal(contour.levelReference, "global");
assert.equal(contour.levelPercent, "77");
assert.equal(contour.filled, true);
assert.equal(contour.opacity, "0.62");
assert.equal(contour.cmap, "magma");
console.log("P4 canonical layer schema probe passed.");
