#!/usr/bin/env node
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import ts from "typescript";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
async function loadTs(relativePath) {
  const source = fs.readFileSync(path.join(root, relativePath), "utf8");
  const output = ts.transpileModule(source, {
    compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 }
  }).outputText;
  return import(`data:text/javascript;base64,${Buffer.from(output).toString("base64")}`);
}

const { predictResolution } = await loadTs("src/timeResolution.ts");
const day = 1 / 86400;
const axis = [0, day, 2 * day];
assert.equal(predictResolution(axis, 0.5 * day, 60, "nearest").resolvedIndex, 0, "nearest exact tie must choose lower index");
assert.equal(predictResolution([0, day, day, 2 * day], 1.1 * day, 60, "previous").resolvedIndex, 2, "previous duplicate must choose greatest matching index");
assert.equal(predictResolution([0, day, day, 2 * day], 0.9 * day, 60, "next").resolvedIndex, 1, "next duplicate must choose lowest matching index");
assert.equal(predictResolution(axis, -day, 60, "previous").unavailable, true, "previous before first sample must be unavailable");
assert.equal(predictResolution(axis, 3 * day, 60, "next").unavailable, true, "next after last sample must be unavailable");
assert.equal(predictResolution(axis, 0.5 * day, 0.25, "nearest").unavailable, true, "tolerance must produce unavailable");
assert.equal(predictResolution(axis, 0.5 * day, 60, "nearest").resolvedMjd, 0, "nearest prediction should expose resolved MJD");

const {
  areFramesCached,
  clearFrameCache,
  hasFrameResult,
  loadFrame,
  prefetchFrames,
  requestKey
} = await loadTs("src/frameScheduler.ts");
const base = { sessionId: "s", sourceId: "radio", operation: "none", reference: "previous", kind: "image", displayParams: { samplingPolicy: "nearest", maxOffsetSeconds: 1 } };
const nearestKey = requestKey({ ...base, resolvedIndex: 0 });
const nextKey = requestKey({ ...base, resolvedIndex: 1 });
const policyKey = requestKey({ ...base, resolvedIndex: 0, displayParams: { samplingPolicy: "next", maxOffsetSeconds: 1 } });
assert.notEqual(nearestKey, nextKey, "different native resolved indexes must not alias");
assert.notEqual(nearestKey, policyKey, "different sampling policies must not alias when rendered identity can differ");

const request = (key, url = `https://example.test/${key}`) => ({
  url,
  key: `${key}|cursor=0`,
  bitmapKey: key,
  predictedResolution: { resolvedIndex: 0, resolvedMjd: 0, offsetSeconds: 0, unavailable: false }
});
globalThis.createImageBitmap = async (blob) => {
  const [width, height] = (await blob.text()).split("x").map(Number);
  return { width, height, close() {} };
};

let active = 0;
let peak = 0;
globalThis.fetch = async () => {
  active += 1;
  peak = Math.max(peak, active);
  await new Promise((resolve) => setTimeout(resolve, 5));
  active -= 1;
  return new Response("2x2", { status: 200, headers: { "Content-Type": "image/png" } });
};
const concurrentRequests = Array.from({ length: 8 }, (_, index) => request(`concurrent-${index}`));
await prefetchFrames(concurrentRequests, new AbortController().signal);
assert.equal(peak, 6, "lookahead prefetch must use bounded concurrency of six");
assert.equal(areFramesCached(concurrentRequests), true, "prefetched frame identities must share the decoded cache");

clearFrameCache();
globalThis.fetch = async (url) => new Response(url.endsWith("large-a") || url.endsWith("large-b") ? "10000x10000" : "2x2", { status: 200 });
await loadFrame(request("large-a"), new AbortController().signal);
await loadFrame(request("large-b"), new AbortController().signal);
assert.equal(hasFrameResult("large-a"), false, "byte-budgeted LRU must evict the oldest decoded bitmap");
assert.equal(hasFrameResult("large-b"), true, "byte-budgeted LRU must retain the newest decoded bitmap");
clearFrameCache();
console.log("Scheduler/resolution probe passed.");
