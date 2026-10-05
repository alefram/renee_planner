import { useEffect, useRef } from "react";
import { orderColor } from "../colors";
import type { CameraPose } from "../types";
import { Section } from "./ui";

export function PoseList({ poses, total, selectedId, onSelect }: {
  poses: CameraPose[];
  total: number;
  selectedId: number | null;
  onSelect: (id: number | null) => void;
}) {
  const selectedRow = useRef<HTMLTableRowElement | null>(null);
  // Braces: the effect must not return scrollIntoView's result (a Promise in recent
  // Chrome), or React calls it as the cleanup on the next selection and the app crashes.
  useEffect(() => {
    selectedRow.current?.scrollIntoView({ block: "nearest" });
  }, [selectedId]);
  return (
    <Section title={`Poses (${poses.length})`}>
      <div className="max-h-80 overflow-y-auto rounded-md border border-slate-200 dark:border-slate-800">
        <table className="w-full text-left text-xs tabular-nums">
          <thead className="sticky top-0 bg-slate-100 text-slate-500 dark:bg-slate-900 dark:text-slate-400">
            <tr>
              <th className="px-2 py-1">#</th>
              <th className="px-2 py-1">Section</th>
              <th className="px-2 py-1 text-right">Dist.</th>
              <th className="px-2 py-1 text-right">Tilt</th>
              <th className="px-2 py-1 text-right">Points</th>
            </tr>
          </thead>
          <tbody>
            {poses.map((pose) => {
              const selected = pose.id === selectedId;
              return (
                <tr
                  key={pose.id}
                  ref={selected ? selectedRow : undefined}
                  onClick={() => onSelect(selected ? null : pose.id)}
                  className={`cursor-pointer border-t border-slate-100 dark:border-slate-800 ${selected ? "bg-amber-100 dark:bg-amber-900/40" : "hover:bg-slate-50 dark:hover:bg-slate-800/60"}`}
                >
                  <td className="px-2 py-1">
                    <span className="mr-1.5 inline-block h-2 w-2 rounded-full" style={{ background: orderColor(pose.id, total) }} />
                    {pose.id}
                  </td>
                  <td className="px-2 py-1">{pose.section}</td>
                  <td className="px-2 py-1 text-right">{pose.standoff_m.toFixed(2)}</td>
                  <td className="px-2 py-1 text-right">{pose.tilt_deg.toFixed(0)}°</td>
                  <td className="px-2 py-1 text-right">{pose.covered}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </Section>
  );
}

export function PoseDetail({ pose }: { pose: CameraPose }) {
  const fmt = (v: number[]) => v.map((x) => x.toFixed(3)).join(", ");
  return (
    <Section title={`Pose ${pose.id} · ${pose.name}`}>
      <dl className="grid grid-cols-[auto,1fr] gap-x-3 gap-y-1 text-xs tabular-nums">
        <dt className="text-slate-500">position</dt>
        <dd className="font-mono">[{fmt(pose.position)}]</dd>
        <dt className="text-slate-500">orientation</dt>
        <dd className="font-mono">[{fmt(pose.orientation)}]</dd>
        <dt className="text-slate-500">look_at</dt>
        <dd className="font-mono">[{fmt(pose.look_at)}]</dd>
        <dt className="text-slate-500">lap</dt>
        <dd>{pose.arc_m.toFixed(2)} m</dd>
      </dl>
    </Section>
  );
}
