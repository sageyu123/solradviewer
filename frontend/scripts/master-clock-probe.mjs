#!/usr/bin/env node
/* Regression probe for Master clock switches clearing a stale custom Δt. */
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const app = fs.readFileSync(path.join(root, "src/App.tsx"), "utf8");
const start = app.indexOf("function setMasterSource(nextSourceId: string)");
const end = app.indexOf("function applyLayerPatch", start);
assert.ok(start >= 0 && end > start, "Master clock handler missing");

const handler = app.slice(start, end);
assert.equal((handler.match(/setTimeStepSeconds\(0\)/g) ?? []).length, 2, "both Master clock paths must restore native cadence");
assert.match(handler, /if \(!nextTimes\.length\)[\s\S]*setTimeStepSeconds\(0\)/, "fallback Master clock path leaves custom Δt active");
assert.match(handler, /setMasterSourceId\(nextSourceId\);\s*setTimeStepSeconds\(0\)/, "successful Master clock switch leaves custom Δt active");
console.log("Master clock cadence reset probe passed.");
