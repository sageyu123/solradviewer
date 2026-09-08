export type SamplingPolicy = "nearest" | "previous" | "next";

export type PredictedResolution = {
  resolvedIndex: number | null;
  resolvedMjd: number | null;
  offsetSeconds: number | null;
  unavailable: boolean;
};

/** Mirror backend resolve_time_index, including duplicate/tie behavior. */
export function predictResolution(
  values: number[],
  requestedMjd: number,
  maxOffsetSeconds: number,
  samplingPolicy: SamplingPolicy = "nearest",
): PredictedResolution {
  const finite = values
    .map((value, index) => ({ value, index }))
    .filter((entry) => Number.isFinite(entry.value));
  if (!finite.length || !Number.isFinite(requestedMjd) || !Number.isFinite(maxOffsetSeconds) || maxOffsetSeconds < 0) {
    return { resolvedIndex: null, resolvedMjd: null, offsetSeconds: null, unavailable: true };
  }
  let candidates = finite;
  if (samplingPolicy === "previous") {
    candidates = finite.filter((entry) => entry.value <= requestedMjd);
    if (!candidates.length) return { resolvedIndex: null, resolvedMjd: null, offsetSeconds: null, unavailable: true };
    const bestValue = Math.max(...candidates.map((entry) => entry.value));
    candidates = candidates.filter((entry) => entry.value === bestValue);
    candidates.sort((left, right) => right.index - left.index);
  } else if (samplingPolicy === "next") {
    candidates = finite.filter((entry) => entry.value >= requestedMjd);
    if (!candidates.length) return { resolvedIndex: null, resolvedMjd: null, offsetSeconds: null, unavailable: true };
    const bestValue = Math.min(...candidates.map((entry) => entry.value));
    candidates = candidates.filter((entry) => entry.value === bestValue);
    candidates.sort((left, right) => left.index - right.index);
  } else {
    let bestDistance = Infinity;
    candidates = finite.filter((entry) => {
      const distance = Math.abs(entry.value - requestedMjd);
      if (distance < bestDistance) bestDistance = distance;
      return true;
    }).filter((entry) => Math.abs(entry.value - requestedMjd) === bestDistance)
      .sort((left, right) => left.index - right.index);
  }
  const best = candidates[0];
  const offsetSeconds = (best.value - requestedMjd) * 86400;
  return {
    resolvedIndex: best.index,
    resolvedMjd: best.value,
    offsetSeconds,
    unavailable: Math.abs(offsetSeconds) > maxOffsetSeconds,
  };
}
