import type { ReactNode } from "react";

export function Section({ title, children, right }: { title: string; children: ReactNode; right?: ReactNode }) {
  return (
    <section className="border-b border-slate-200 px-4 py-3 dark:border-slate-800">
      <div className="mb-2 flex items-center justify-between">
        <h2 className="text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">{title}</h2>
        {right}
      </div>
      {children}
    </section>
  );
}

export function Toggle({ label, checked, onChange, swatch }: { label: string; checked: boolean; onChange: (v: boolean) => void; swatch?: string }) {
  return (
    <label className="flex cursor-pointer select-none items-center gap-2 text-sm">
      <input type="checkbox" className="h-4 w-4 accent-sky-600" checked={checked} onChange={(e) => onChange(e.target.checked)} />
      {swatch && <span className="h-3 w-3 rounded-sm" style={{ background: swatch }} />}
      <span>{label}</span>
    </label>
  );
}

export const selectClass =
  "w-full rounded-md border border-slate-300 bg-white px-2 py-1.5 text-sm dark:border-slate-700 dark:bg-slate-900";
export const buttonClass =
  "rounded-md border border-slate-300 px-2.5 py-1 text-sm hover:bg-slate-100 disabled:opacity-40 dark:border-slate-700 dark:hover:bg-slate-800";
