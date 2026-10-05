import { Html } from "@react-three/drei";
import type { ThreeEvent } from "@react-three/fiber";
import { useMemo } from "react";
import * as THREE from "three";
import { orderColor } from "../colors";
import type { CameraPose, Quat, Vec3 } from "../types";

/** Frustum lines of the optical frame (+Z forward) out to `depth`, for a [h, v] field of view. */
export function frustumGeometry(fovDeg: [number, number], depth: number): THREE.BufferGeometry {
  const x = Math.tan((fovDeg[0] * Math.PI) / 360) * depth;
  const y = Math.tan((fovDeg[1] * Math.PI) / 360) * depth;
  const corners: Vec3[] = [[-x, -y, depth], [x, -y, depth], [x, y, depth], [-x, y, depth]];
  const vertices: number[] = [];
  corners.forEach((c, i) => {
    const next = corners[(i + 1) % 4];
    vertices.push(0, 0, 0, ...c, ...c, ...next);
  });
  // Image "up" marker (-Y in the optical frame) on the far rectangle.
  vertices.push(-x * 0.3, -y, depth, 0, -y * 1.3, depth, 0, -y * 1.3, depth, x * 0.3, -y, depth);
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(vertices, 3));
  return geometry;
}

export function Frustum({ position, orientation, geometry, color, emphasis = false, onClick }: {
  position: Vec3;
  orientation: Quat;
  geometry: THREE.BufferGeometry;
  color: string;
  emphasis?: boolean;
  onClick?: (event: ThreeEvent<MouseEvent>) => void;
}) {
  const quaternion = useMemo(() => new THREE.Quaternion(...orientation), [orientation]);
  return (
    <group position={position} quaternion={quaternion}>
      <lineSegments geometry={geometry}>
        <lineBasicMaterial color={color} transparent opacity={emphasis ? 1 : 0.75} />
      </lineSegments>
      <mesh onClick={onClick} scale={emphasis ? 1.6 : 1}>
        <sphereGeometry args={[0.03, 12, 8]} />
        <meshBasicMaterial color={color} />
      </mesh>
      {emphasis && <axesHelper args={[0.15]} />}
    </group>
  );
}

export function CameraPoses({ poses, total, fovDeg, depth, selectedId, onSelect, showLabels, ghostColor }: {
  poses: CameraPose[];
  total: number;
  fovDeg: [number, number];
  depth: number;
  selectedId?: number | null;
  onSelect?: (id: number) => void;
  showLabels: boolean;
  ghostColor?: string; // compare mode: one flat colour, not clickable
}) {
  const geometry = useMemo(() => frustumGeometry(fovDeg, depth), [fovDeg[0], fovDeg[1], depth]);
  return (
    <group>
      {poses.map((pose) => {
        const selected = pose.id === selectedId;
        const color = ghostColor ?? (selected ? "#ffd60a" : orderColor(pose.id, total));
        return (
          <group key={pose.id}>
            <Frustum
              position={pose.position}
              orientation={pose.orientation}
              geometry={geometry}
              color={color}
              emphasis={selected}
              onClick={onSelect ? (event) => (event.stopPropagation(), onSelect(pose.id)) : undefined}
            />
            {(showLabels || selected) && !ghostColor && (
              // DOM labels: no font download (the robot may be offline).
              <Html position={[pose.position[0], pose.position[1], pose.position[2] + 0.08]} center zIndexRange={[10, 0]}>
                <span className="pointer-events-none select-none rounded bg-black/60 px-1 text-[10px] font-semibold" style={{ color }}>
                  {pose.id}
                </span>
              </Html>
            )}
          </group>
        );
      })}
    </group>
  );
}
