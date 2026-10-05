import type { Coverage, Vec3 } from "./types";

export interface PickedLink {
  link: string;
  section: string;
  point: Vec3;
  time: string;
}

/** Index of the surface point closest to `p` and its distance (brute force: one click, ~100k points). */
export function nearestPoint(points: number[], p: Vec3): { index: number; distance: number } {
  let best = -1;
  let bestDistance = Infinity;
  for (let i = 0; i < points.length; i += 3) {
    const dx = points[i] - p[0], dy = points[i + 1] - p[1], dz = points[i + 2] - p[2];
    const d = dx * dx + dy * dy + dz * dz;
    if (d < bestDistance) {
      bestDistance = d;
      best = i / 3;
    }
  }
  return { index: best, distance: Math.sqrt(bestDistance) };
}

/** A click on the mesh far from every surface point: a part outside the trajectory's target. */
export function outsideTarget(point: Vec3): PickedLink {
  return { link: "(no inspected surface here: outside the target, or a bottom/hidden face)", section: "–", point, time: new Date().toLocaleTimeString() };
}

/** The link and section of surface point `index`, or null if the trajectory has no part ids. */
export function linkOf(coverage: Coverage, index: number): PickedLink | null {
  if (index < 0 || !coverage.part_ids || !coverage.part_names) return null;
  const point: Vec3 = [coverage.points[3 * index], coverage.points[3 * index + 1], coverage.points[3 * index + 2]];
  return {
    link: coverage.part_names[coverage.part_ids[index]],
    section: coverage.section_names[coverage.section_ids[index]],
    point,
    time: new Date().toLocaleTimeString(),
  };
}
