import { Line } from "@react-three/drei";
import { useMemo } from "react";
import { orderColor } from "../colors";
import type { CameraPose, Vec3 } from "../types";

/** The visiting order: a line through the camera positions, blue (first) to red (last). */
export function Route({ poses }: { poses: CameraPose[] }) {
  const { points, colors } = useMemo(() => {
    const colors = poses.map((p) => orderColor(p.id, poses.length));
    return { points: poses.map((p) => p.position), colors };
  }, [poses]);
  if (points.length < 2) return null;
  return <Line points={points} vertexColors={colors.map(cssToRgb)} lineWidth={2} />;
}

/** The base lane around the machine (plan view, on the floor). */
export function Lane({ laneXy }: { laneXy: number[] }) {
  const points = useMemo(() => {
    const list: Vec3[] = [];
    for (let i = 0; i < laneXy.length; i += 2) list.push([laneXy[i], laneXy[i + 1], 0.01]);
    if (list.length) list.push(list[0]);
    return list;
  }, [laneXy]);
  if (points.length < 3) return null;
  return <Line points={points} color="#7c8799" lineWidth={1} dashed dashSize={0.15} gapSize={0.1} />;
}

function cssToRgb(color: string): [number, number, number] {
  const match = /hsl\((\d+), (\d+)%, (\d+)%\)/.exec(color);
  if (!match) return [1, 1, 1];
  const [h, s, l] = [Number(match[1]) / 360, Number(match[2]) / 100, Number(match[3]) / 100];
  const k = (n: number) => (n + h * 12) % 12;
  const a = s * Math.min(l, 1 - l);
  const f = (n: number) => l - a * Math.max(-1, Math.min(k(n) - 3, 9 - k(n), 1));
  return [f(0), f(8), f(4)];
}
