import { GizmoHelper, GizmoViewport, Grid, OrbitControls } from "@react-three/drei";
import { Canvas, useThree } from "@react-three/fiber";
import { type ReactNode, useEffect } from "react";
import * as THREE from "three";
import type { OrbitControls as OrbitControlsImpl } from "three-stdlib";
import type { Vec3 } from "../types";

// The map frame is Z-up (ROS); make three.js agree.
THREE.Object3D.DEFAULT_UP.set(0, 0, 1);

/** Points the view at `center` from the south-east, `distance` away, when `center` changes. */
function FitView({ center, distance }: { center: Vec3 | null; distance: number }) {
  const { camera, controls } = useThree();
  useEffect(() => {
    if (!center) return;
    const [x, y, z] = center;
    camera.position.set(x + distance * 0.7, y - distance * 0.7, z + distance * 0.6);
    const orbit = controls as unknown as OrbitControlsImpl | null;
    orbit?.target.set(x, y, z);
    orbit?.update();
  }, [center?.[0], center?.[1], center?.[2], distance, camera, controls]);
  return null;
}

export function Scene({ center, extent, children }: { center: Vec3 | null; extent: number; children: ReactNode }) {
  return (
    <Canvas
      camera={{ fov: 50, near: 0.02, far: 500, up: [0, 0, 1], position: [8, -8, 6] }}
      dpr={[1, 2]}
      // Clicking a surface point picks it within 2 cm (the default 1 m picks everything).
      raycaster={{ params: { Points: { threshold: 0.02 } } as THREE.RaycasterParameters }}
    >
      <color attach="background" args={["#14171c"]} />
      <ambientLight intensity={0.6} />
      <directionalLight position={[10, -6, 12]} intensity={1.2} />
      <directionalLight position={[-8, 10, 6]} intensity={0.4} />
      <Grid
        rotation={[Math.PI / 2, 0, 0]}
        args={[60, 60]}
        cellSize={0.5}
        sectionSize={2}
        cellColor="#2a2f38"
        sectionColor="#3c4452"
        infiniteGrid
        fadeDistance={60}
      />
      <axesHelper args={[0.5]} />
      <OrbitControls makeDefault enableDamping />
      <FitView center={center} distance={Math.max(4, extent * 1.3)} />
      <GizmoHelper alignment="bottom-right" margin={[70, 70]}>
        <GizmoViewport labelColor="white" axisHeadScale={0.9} />
      </GizmoHelper>
      {children}
    </Canvas>
  );
}
