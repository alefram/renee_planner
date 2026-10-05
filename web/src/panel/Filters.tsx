import { COLORS, sectionColor } from "../colors";
import { Section, Toggle } from "./ui";

export interface Layers {
  machine: boolean;
  coverage: boolean;
  frustums: boolean;
  route: boolean;
  lane: boolean;
  labels: boolean;
  live: boolean;
}

const rgb = (c: readonly number[]) => `rgb(${c.map((v) => Math.round(v * 255)).join(",")})`;

export function Filters({ layers, setLayers, sections, visibleSections, setVisibleSections, depth, setDepth, opacity, setOpacity, liveAvailable }: {
  layers: Layers;
  setLayers: (l: Layers) => void;
  sections: string[];
  visibleSections: Set<number>;
  setVisibleSections: (s: Set<number>) => void;
  depth: number;
  setDepth: (d: number) => void;
  opacity: number;
  setOpacity: (o: number) => void;
  liveAvailable: boolean;
}) {
  const toggle = (key: keyof Layers) => (v: boolean) => setLayers({ ...layers, [key]: v });
  const toggleSection = (index: number, on: boolean) => {
    const next = new Set(visibleSections);
    if (on) next.add(index);
    else next.delete(index);
    setVisibleSections(next);
  };
  return (
    <>
      <Section title="Layers">
        <div className="grid grid-cols-2 gap-1.5">
          <Toggle label="Machine" checked={layers.machine} onChange={toggle("machine")} />
          <Toggle label="Coverage" checked={layers.coverage} onChange={toggle("coverage")} />
          <Toggle label="Frustums" checked={layers.frustums} onChange={toggle("frustums")} />
          <Toggle label="Route" checked={layers.route} onChange={toggle("route")} />
          <Toggle label="Base lane" checked={layers.lane} onChange={toggle("lane")} />
          <Toggle label="Numbers" checked={layers.labels} onChange={toggle("labels")} />
          {liveAvailable && <Toggle label="Live camera" checked={layers.live} onChange={toggle("live")} swatch={COLORS.live} />}
        </div>
        <label className="mt-3 block text-sm">
          <span className="text-slate-500 dark:text-slate-400">Frustum depth {depth.toFixed(2)} m</span>
          <input type="range" min={0.1} max={1.5} step={0.05} value={depth} onChange={(e) => setDepth(Number(e.target.value))} className="w-full accent-sky-600" />
        </label>
        <label className="block text-sm">
          <span className="text-slate-500 dark:text-slate-400">Machine opacity {Math.round(opacity * 100)}%</span>
          <input type="range" min={0.1} max={1} step={0.05} value={opacity} onChange={(e) => setOpacity(Number(e.target.value))} className="w-full accent-sky-600" />
        </label>
        <div className="mt-2 flex flex-wrap gap-x-3 gap-y-1 text-xs text-slate-500 dark:text-slate-400">
          <span><span className="mr-1 inline-block h-2 w-2 rounded-full" style={{ background: rgb(COLORS.covered) }} />1 view</span>
          <span><span className="mr-1 inline-block h-2 w-2 rounded-full" style={{ background: rgb(COLORS.overlap) }} />2+ views</span>
          <span><span className="mr-1 inline-block h-2 w-2 rounded-full" style={{ background: rgb(COLORS.uncovered) }} />not covered</span>
          <span><span className="mr-1 inline-block h-2 w-2 rounded-full" style={{ background: rgb(COLORS.unreachable) }} />unreachable</span>
          <span><span className="mr-1 inline-block h-2 w-2 rounded-full" style={{ background: rgb(COLORS.selected) }} />selected pose</span>
        </div>
      </Section>
      <Section title="Sections">
        <div className="grid grid-cols-2 gap-1.5">
          {sections.map((name, i) => (
            <Toggle key={name} label={name} swatch={sectionColor(i)} checked={visibleSections.has(i)} onChange={(v) => toggleSection(i, v)} />
          ))}
        </div>
      </Section>
    </>
  );
}
