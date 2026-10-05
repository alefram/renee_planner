import { useGLTF } from "@react-three/drei";
import type { ThreeEvent } from "@react-three/fiber";
import { useEffect, useMemo } from "react";
import * as THREE from "three";

/** The machine mesh (decimated CAD or the reconstruction), already in the map frame. */
export function Machine({ url, opacity, onPick }: { url: string; opacity: number; onPick?: (point: [number, number, number]) => void }) {
  const { scene } = useGLTF(url);
  const object = useMemo(() => scene.clone(true), [scene]);
  useEffect(() => {
    object.traverse((child) => {
      const mesh = child as THREE.Mesh;
      if (!mesh.isMesh) return;
      const material = new THREE.MeshStandardMaterial({
        vertexColors: Boolean(mesh.geometry.getAttribute("color")),
        color: mesh.geometry.getAttribute("color") ? "#ffffff" : "#8c96a8",
        roughness: 0.8,
        metalness: 0.1,
        side: THREE.DoubleSide,
        transparent: opacity < 1,
        opacity,
        depthWrite: opacity >= 1,
      });
      mesh.material = material;
    });
  }, [object, opacity]);
  const click = (event: ThreeEvent<MouseEvent>) => {
    if (!onPick || event.delta > 4) return; // a drag (orbit), not a click
    event.stopPropagation();
    onPick([event.point.x, event.point.y, event.point.z]);
  };
  return <primitive object={object} onClick={click} />;
}
