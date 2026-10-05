// Colours shared by the scene and the panel.

/** Pose colour by its place in the order: blue (first) to red (last). */
export function orderColor(index: number, count: number): string {
  const hue = 220 - (220 * index) / Math.max(1, count - 1);
  return `hsl(${hue.toFixed(0)}, 80%, 55%)`;
}

export const SECTION_COLORS = ["#5b8ccc", "#d98c40", "#73b373", "#bf6699", "#999999", "#ccbf59", "#66b3bf"];
export const sectionColor = (index: number) => SECTION_COLORS[index % SECTION_COLORS.length];

export const COLORS = {
  uncovered: [0.9, 0.2, 0.2] as const, // coverable but not seen by the chosen poses
  unreachable: [0.35, 0.35, 0.38] as const, // no candidate sees it
  covered: [0.2, 0.75, 0.35] as const,
  overlap: [0.2, 0.5, 0.95] as const, // seen by 2 or more poses
  selected: [1.0, 0.85, 0.1] as const, // seen by the selected pose
  live: "#e040fb",
  compare: "#8a8a8a",
};
