#!/usr/bin/env node
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import ts from "typescript";

const filename = fileURLToPath(import.meta.url);
const sourcePath = path.resolve(path.dirname(filename), "../src/radioColormaps.ts");
const source = fs.readFileSync(sourcePath, "utf8");
for (const token of ["parula", "inferno", "rdylbu", "sampleColormap", "ColormapDirection", "backendColormap", "parula_r"]) {
  assert.ok(source.includes(token), `palette registry token missing: ${token}`);
}
assert.ok(source.indexOf("#352a87") < source.indexOf("#f9fb0e"), "Parula scalar endpoints changed");
assert.ok(source.indexOf("#000004") < source.indexOf("#fcffa4"), "Inferno scalar endpoints changed");
assert.ok(source.indexOf("#a50026") < source.indexOf("#313695"), "RdYlBu frequency endpoints changed");
assert.ok(source.includes('direction === "frequency"'), "frequency direction is not explicit");
const transpiled = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 }
}).outputText;
const registry = await import(`data:text/javascript;base64,${Buffer.from(transpiled).toString("base64")}`);
assert.equal(registry.normalizeColormap("RDYlBU_R"), "rdylbu");
assert.equal(registry.backendColormap("Parula", "frequency"), "parula_r");
assert.equal(registry.backendColormap("Inferno", "frequency"), "inferno_r");
assert.equal(registry.backendColormap("Viridis", "frequency"), "viridis_r");
assert.equal(registry.backendColormap("RdYlBu", "frequency"), "RdYlBu");
assert.equal(registry.sampleColormap("Parula", 0, 11, "frequency"), "#f9fb0e");
assert.equal(registry.sampleColormap("Parula", 10, 11, "frequency"), "#352a87");
assert.equal(registry.sampleColormap("RdYlBu", 0, 11, "frequency"), "#a50026");
assert.equal(registry.sampleColormap("RdYlBu", 10, 11, "frequency"), "#313695");
assert.equal(registry.sampleColormap("Viridis", 0, 11, "frequency"), "#fde725");
assert.equal(registry.sampleColormap("Viridis", 10, 11, "frequency"), "#440154");
assert.equal(registry.sampleColormap("Inferno", 0, 11, "frequency"), "#fcffa4");
assert.equal(registry.sampleColormap("Inferno", 10, 11, "frequency"), "#000004");
const app = fs.readFileSync(path.resolve(path.dirname(filename), "../src/App.tsx"), "utf8");
assert.ok(app.includes("1 - index / Math.max(1, stops.length - 1)"), "vertical colorbar orientation changed");
console.log("P5 colormap registry probe passed.");
