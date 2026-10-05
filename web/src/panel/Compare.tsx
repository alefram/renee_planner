import type { TrajectorySummary } from "../types";
import { Section, buttonClass, selectClass } from "./ui";

const label = (t: TrajectorySummary) =>
  t.error ? `${t.name} (unreadable)` : `${t.name} · ${t.mode} · ${t.poses} poses · ${((t.coverage ?? 0) * 100).toFixed(0)}%`;

/** Pick the trajectory to show and, optionally, a second one drawn in grey for comparison. */
export function Compare({ items, current, compare, onCurrent, onCompare, onRefresh }: {
  items: TrajectorySummary[];
  current: string | null;
  compare: string | null;
  onCurrent: (name: string) => void;
  onCompare: (name: string | null) => void;
  onRefresh: () => void;
}) {
  return (
    <Section title="Trajectory" right={<button className={buttonClass} onClick={onRefresh}>Refresh</button>}>
      <select className={selectClass} value={current ?? ""} onChange={(e) => onCurrent(e.target.value)}>
        {items.length === 0 && <option value="">No trajectories yet</option>}
        {items.map((t) => (
          <option key={t.name} value={t.name} disabled={Boolean(t.error)}>{label(t)}</option>
        ))}
      </select>
      <label className="mt-2 block text-xs text-slate-500 dark:text-slate-400">Compare with</label>
      <select className={selectClass} value={compare ?? ""} onChange={(e) => onCompare(e.target.value || null)}>
        <option value="">—</option>
        {items.filter((t) => t.name !== current && !t.error).map((t) => (
          <option key={t.name} value={t.name}>{label(t)}</option>
        ))}
      </select>
    </Section>
  );
}
