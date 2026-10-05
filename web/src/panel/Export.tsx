import { yamlUrl } from "../api";
import type { CameraPose } from "../types";
import { Section, buttonClass } from "./ui";

export function Export({ name, selected }: { name: string; selected: CameraPose | null }) {
  const copy = () => selected && navigator.clipboard?.writeText(JSON.stringify(selected, null, 2));
  return (
    <Section title="Export">
      <div className="flex flex-wrap gap-2">
        <a className={buttonClass} href={yamlUrl(name)} download={`${name}.yaml`}>Download YAML</a>
        <button className={buttonClass} disabled={!selected} onClick={copy}>Copy selected pose</button>
      </div>
    </Section>
  );
}
