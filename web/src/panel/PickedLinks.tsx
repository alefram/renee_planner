import type { PickedLink } from "../picking";
import { Section, buttonClass } from "./ui";

/** Log of the links clicked in the scene (newest first): the names target.parts takes. */
export function PickedLinks({ picks, onClear, missingParts }: { picks: PickedLink[]; onClear: () => void; missingParts: boolean }) {
  const copy = (text: string) => navigator.clipboard?.writeText(text);
  return (
    <Section title="Picked links" right={picks.length > 0 && <button className={buttonClass} onClick={onClear}>Clear</button>}>
      {missingParts ? (
        <p className="text-xs text-amber-600">This trajectory has no link per point: generate it again to pick links.</p>
      ) : picks.length === 0 ? (
        <p className="text-xs text-slate-500">Click the machine or a surface point to see its link.</p>
      ) : (
        <ul className="space-y-1 text-xs">
          {picks.map((pick, i) => (
            <li key={`${pick.time}-${i}`} className="flex items-center justify-between gap-2 rounded border border-slate-200 px-2 py-1 dark:border-slate-800">
              <span className="min-w-0">
                <span className="font-mono font-semibold">{pick.link}</span>
                <span className="text-slate-500"> · {pick.section} · {pick.time}</span>
                <span className="block font-mono text-[10px] text-slate-500">[{pick.point.map((v) => v.toFixed(3)).join(", ")}]</span>
              </span>
              <button className={buttonClass} onClick={() => copy(pick.link.replace(/_link$/, ""))} title="Copy for target.parts">
                Copy
              </button>
            </li>
          ))}
        </ul>
      )}
    </Section>
  );
}
