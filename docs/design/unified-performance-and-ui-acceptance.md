# Unified performance and UI acceptance

Date: 2026-08-08

This document records the final acceptance evidence for the unified timeline,
layered panels, scheduler/cache, colormap, and accessibility work. It is an
evidence record, not a replacement for the implementation plan in
[`unified-performance-and-ui-plan.md`](unified-performance-and-ui-plan.md).

## Launch and dataset

The dataset-key launcher was verified with:

```text
./run_app.sh 20250328
```

The mounted session reported 340 AIA frames, 600 radio frames, and 5,099
spectrogram columns. The radio master slider covered native positions 0..599.

## Layered-panel visual acceptance

The tested composition was:

- left: AIA base, previous-frame ratio, plus radio contours;
- right: AIA original, plus radio contours.

Both panels displayed the expected WCS-aligned imagery, contour overlays, and
frequency colorbars. The radio color convention was visibly warm at low
frequency and cool at high frequency.

## Timeline and request traces

An exact isolated 30-step trace produced 30 radio requests with 30 unique
resolved identities, 2 AIA requests with 2 unique identities, and zero
duplicates.

A live 50 fps run traversed positions 0→599 in 27.415 s. A monotonic browser
observer saw 592 distinct positions; 8 positions were not observed because the
observer polled every 15 ms. The endpoint was reached. This observation must
not be read as proof that every intermediate position was painted to the
browser.

For uncached endpoint samples 100–149:

| Measurement | Result |
| --- | ---: |
| Frames sampled | 50 |
| AIA resolved changes | 5 |
| Settled latency, median | 21.22 ms |
| Settled latency, p90 | 87.22 ms |
| Radio-only latency, median | 20.81 ms |
| Cold outlier maximum | 1,223.99 ms |

For comparison, prior baselines were approximately 204 ms for unique/full-cube
EOVSA requests and 144 ms for cold AIA requests.

## Verification summary

- Backend suite: 42 tests passed, with 24 subtests.
- P4 canonical-layer probe: passed.
- P5 colormap registry probe: passed.
- P6 accessibility/status/guard probe: passed.
- Scheduler/time-resolution probe: passed.
- Frontend production build (`npm run build`): passed.

The final policy-aware cache-identity regression gate covers nearest,
previous, and next sampling, including tie/duplicate selection, edge
unavailability, tolerance-based 204 prediction, and non-aliasing native
resolved identities.

The measurements above were taken against the launched 20250328 session. No
claim is made that a 15 ms browser observer captures every frame; request-level
traces and endpoint reachability are the authoritative checks for traversal.
