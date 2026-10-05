import { useMemo } from "react";
import { COLORS } from "../colors";
import type { Quat, Vec3 } from "../types";
import { Frustum, frustumGeometry } from "./CameraPoses";

/** Where the robot's camera is now (from /tf over rosbridge). */
export function LiveCamera({ pose, fovDeg, depth }: { pose: { position: Vec3; orientation: Quat }; fovDeg: [number, number]; depth: number }) {
  const geometry = useMemo(() => frustumGeometry(fovDeg, depth), [fovDeg[0], fovDeg[1], depth]);
  return <Frustum position={pose.position} orientation={pose.orientation} geometry={geometry} color={COLORS.live} emphasis />;
}
