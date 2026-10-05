import type { Trajectory } from "../types";
import { Section } from "./ui";

const percent = (v: number | undefined) => (v === undefined ? "–" : `${(v * 100).toFixed(1)}%`);

export function Stats({ trajectory }: { trajectory: Trajectory }) {
  const c = trajectory.coverage;
  const rows: [string, string][] = [
    ["Poses", String(trajectory.poses.length)],
    ["Coverage (all points)", percent(c.ratio)],
    ["Coverage (reachable)", percent(c.coverable_ratio)],
    ["Uncovered points", `${c.uncovered_points} / ${c.surface_points}`],
    ["Candidates", String(c.candidates)],
    ["Stopped by", c.stop_reason],
    ["Lap length", `${Number(trajectory.notes?.lap_length_m ?? 0).toFixed(1)} m`],
    ["Machine pose", trajectory.machine.corrected_by_icp ? "corrected by ICP" : trajectory.machine.pose_source],
    ["Generated", trajectory.generated],
  ];
  return (
    <Section title={`${trajectory.experiment} · ${trajectory.mode}`}>
      <dl className="grid grid-cols-2 gap-x-3 gap-y-1 text-sm">
        {rows.map(([k, v]) => (
          <div key={k} className="contents">
            <dt className="text-slate-500 dark:text-slate-400">{k}</dt>
            <dd className="text-right font-medium tabular-nums">{v}</dd>
          </div>
        ))}
      </dl>
    </Section>
  );
}
