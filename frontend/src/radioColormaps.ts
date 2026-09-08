// Palette stops and frequency convention follow ovrolwa-rfr-corr-app/frontend/src/radioColormaps.ts.
export type ColormapId = "gray" | "gray_r" | "viridis" | "turbo" | "magma" | "coolwarm" | "parula" | "inferno" | "rdylbu" | "aia94" | "aia131" | "aia171" | "aia193" | "aia211" | "aia304" | "aia335";
export type ColormapDirection = "scalar" | "frequency";

export type ColormapOption = {
  id: ColormapId;
  label: string;
};

export const INSTRUMENT_COLORMAP_OPTIONS: readonly ColormapOption[] = [
  { id: "aia94", label: "AIA 94 Å" },
  { id: "aia131", label: "AIA 131 Å" },
  { id: "aia171", label: "AIA 171 Å" },
  { id: "aia193", label: "AIA 193 Å" },
  { id: "aia211", label: "AIA 211 Å" },
  { id: "aia304", label: "AIA 304 Å" },
  { id: "aia335", label: "AIA 335 Å" }
];

const SCALAR_STOPS: Record<ColormapId, readonly string[]> = {
  gray: ["#000000", "#ffffff"],
  gray_r: ["#ffffff", "#000000"],
  viridis: ["#440154", "#482475", "#414487", "#355f8d", "#2a788e", "#21918c", "#22a884", "#44bf70", "#7ad151", "#bddf26", "#fde725"],
  turbo: ["#30123b", "#4559cb", "#3e9bfe", "#19d5cd", "#46f884", "#a4fc3c", "#e1dd37", "#fea431", "#f05b12", "#c32503", "#7a0403"],
  magma: ["#000004", "#140e36", "#3b0f70", "#641a80", "#8c2981", "#b73779", "#de4968", "#f7705c", "#fe9f6d", "#fecf92", "#fcfdbf"],
  coolwarm: ["#3b4cc0", "#7597f6", "#b9d0f9", "#dddddd", "#f5b69b", "#d65244", "#b40426"],
  parula: ["#352a87", "#1e42ba", "#0575b7", "#079bad", "#07b792", "#2ec971", "#7bd151", "#bdd242", "#ecd14d", "#f9ba59", "#f9fb0e"],
  inferno: ["#000004", "#160b39", "#420a68", "#6a176e", "#932667", "#bc3754", "#dd513a", "#f37819", "#fca50a", "#f6d746", "#fcffa4"],
  rdylbu: ["#a50026", "#d62f27", "#f46d43", "#fdad60", "#fee090", "#feffc0", "#e0f3f8", "#aad8e9", "#74add1", "#4574b3", "#313695"],
  aia94: ["#000000", "#ffffff"],
  aia131: ["#000000", "#ffffff"],
  aia171: ["#000000", "#ffffff"],
  aia193: ["#000000", "#ffffff"],
  aia211: ["#000000", "#ffffff"],
  aia304: ["#000000", "#ffffff"],
  aia335: ["#000000", "#ffffff"]
};

export const COLORMAP_OPTIONS: readonly ColormapOption[] = [
  { id: "gray", label: "Gray" },
  { id: "gray_r", label: "Gray R" },
  { id: "viridis", label: "Viridis" },
  { id: "turbo", label: "Turbo" },
  { id: "magma", label: "Magma" },
  { id: "coolwarm", label: "Coolwarm" },
  { id: "parula", label: "Parula" },
  { id: "inferno", label: "Inferno" },
  { id: "rdylbu", label: "RdYlBu" }
];

const ALIASES: Record<string, ColormapId> = {
  Gray: "gray",
  "Gray R": "gray_r",
  Viridis: "viridis",
  Turbo: "turbo",
  Magma: "magma",
  Coolwarm: "coolwarm",
  Parula: "parula",
  Inferno: "inferno",
  RdYlBu: "rdylbu",
  RdYlBu_r: "rdylbu",
  rdylbu_r: "rdylbu",
  parula_r: "parula",
  inferno_r: "inferno",
  viridis_r: "viridis",
  turbo_r: "turbo",
  magma_r: "magma",
  coolwarm_r: "coolwarm"
};

export function normalizeColormap(value: unknown, fallback: ColormapId = "viridis"): ColormapId {
  const text = String(value ?? "");
  const alias = ALIASES[text] ?? ALIASES[text.toLowerCase()];
  return alias ?? (text in SCALAR_STOPS ? text as ColormapId : fallback);
}

export function colormapStops(value: unknown, direction: ColormapDirection = "scalar"): readonly string[] {
  const id = normalizeColormap(value);
  const stops = SCALAR_STOPS[id] ?? SCALAR_STOPS.viridis;
  if (direction === "frequency" && id !== "gray" && id !== "gray_r" && id !== "rdylbu") return [...stops].reverse();
  return stops;
}

export function colormapGradient(value: unknown, direction: ColormapDirection = "scalar"): string {
  return `linear-gradient(90deg, ${colormapStops(value, direction).join(", ")})`;
}

/** Backend colormap id matching the normalized frequency direction. */
export function backendColormap(value: unknown, direction: ColormapDirection = "scalar"): string {
  const id = normalizeColormap(value);
  if (direction !== "frequency" || id === "gray" || id === "gray_r" || id === "rdylbu") {
    return id === "rdylbu" ? "RdYlBu" : id;
  }
  return `${id}_r`;
}

function rgb(color: string): [number, number, number] {
  return [Number.parseInt(color.slice(1, 3), 16), Number.parseInt(color.slice(3, 5), 16), Number.parseInt(color.slice(5, 7), 16)];
}

function hex(red: number, green: number, blue: number): string {
  return `#${[red, green, blue].map((value) => Math.round(value).toString(16).padStart(2, "0")).join("")}`;
}

/** Endpoint-inclusive palette sampling used by contour/frequency legends. */
export function sampleColormap(value: unknown, index: number, count: number, direction: ColormapDirection = "scalar"): string {
  const stops = colormapStops(value, direction);
  if (count <= 1) return stops[0];
  const position = Math.max(0, Math.min(1, index / (count - 1))) * (stops.length - 1);
  const lower = Math.floor(position);
  const upper = Math.min(stops.length - 1, lower + 1);
  if (lower === upper) return stops[lower];
  const fraction = position - lower;
  const start = rgb(stops[lower]);
  const end = rgb(stops[upper]);
  return hex(start[0] + (end[0] - start[0]) * fraction, start[1] + (end[1] - start[1]) * fraction, start[2] + (end[2] - start[2]) * fraction);
}
