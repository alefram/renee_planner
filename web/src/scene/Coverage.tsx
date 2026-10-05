import type { ThreeEvent } from "@react-three/fiber";
import { useEffect, useMemo } from "react";
import * as THREE from "three";
import { COLORS } from "../colors";
import type { Coverage as CoverageData } from "../types";

/** The surface sample points, coloured by views (or by what the selected pose sees). */
export function Coverage({ data, selectedPose, visibleSections, size, onPick }: {
  data: CoverageData;
  selectedPose: number | null;
  visibleSections: Set<number>;
  size: number;
  onPick?: (index: number) => void;
}) {
  const geometry = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(data.points, 3));
    g.setAttribute("color", new THREE.Float32BufferAttribute(new Float32Array(data.points.length), 3));
    return g;
  }, [data]);

  useEffect(() => {
    const colors = geometry.getAttribute("color") as THREE.BufferAttribute;
    const array = colors.array as Float32Array;
    const count = data.view_count.length;
    const seen = new Uint8Array(count);
    if (selectedPose !== null && selectedPose + 1 < data.pose_indptr.length) {
      for (let k = data.pose_indptr[selectedPose]; k < data.pose_indptr[selectedPose + 1]; k++) seen[data.pose_indices[k]] = 1;
    }
    for (let i = 0; i < count; i++) {
      let c: readonly number[];
      if (!visibleSections.has(data.section_ids[i])) c = [0, 0, 0];
      else if (seen[i]) c = COLORS.selected;
      else if (!data.coverable[i]) c = COLORS.unreachable;
      else if (data.view_count[i] === 0) c = COLORS.uncovered;
      else if (data.view_count[i] === 1) c = COLORS.covered;
      else c = COLORS.overlap;
      array.set(c, i * 3);
    }
    colors.needsUpdate = true;
  }, [geometry, data, selectedPose, visibleSections]);

  // Hidden sections: drop their points from the draw (colour black is only a fallback).
  const index = useMemo(() => {
    const list: number[] = [];
    data.section_ids.forEach((s, i) => visibleSections.has(s) && list.push(i));
    return list;
  }, [data, visibleSections]);
  useEffect(() => {
    geometry.setIndex(index);
  }, [geometry, index]);

  return (
    <points
      geometry={geometry}
      onClick={(event: ThreeEvent<MouseEvent>) => {
        if (!onPick || event.delta > 4 || event.index === undefined) return;
        event.stopPropagation();
        onPick(event.index);
      }}
    >
      <pointsMaterial size={size} vertexColors sizeAttenuation />
    </points>
  );
}
