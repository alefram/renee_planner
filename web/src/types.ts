// Shapes of what scan_viewer_server_node serves (see trajectory.py).

export type Vec3 = [number, number, number];
export type Quat = [number, number, number, number]; // [qx, qy, qz, qw]

export interface CameraPose {
  id: number;
  name: string;
  section: string;
  position: Vec3;
  orientation: Quat; // optical frame: +Z forward, +X right, +Y down
  look_at: Vec3;
  covered: number;
  arc_m: number;
  standoff_m: number;
  tilt_deg: number;
}

export interface Trajectory {
  experiment: string;
  mode: "cad" | "sensor";
  frame: string;
  camera: { frame: string; fov_deg: [number, number]; view_distance_m: [number, number]; max_incidence_deg: number };
  coverage: {
    ratio: number;
    coverable_ratio: number;
    surface_points: number;
    coverable_points: number;
    uncovered_points: number;
    views_per_point: number;
    candidates: number;
    stop_reason: string;
    file: string;
  };
  machine: { root_link: string; pose_source: string; corrected_by_icp: boolean; position: Vec3; orientation: Quat; mesh: string };
  poses: CameraPose[];
  generated: string;
  notes: Record<string, unknown>;
}

/** Flat arrays: points[3i..3i+2] is point i. pose_indices[pose_indptr[k]..pose_indptr[k+1]] are the points pose k sees. */
export interface Coverage {
  points: number[];
  normals: number[];
  section_ids: number[];
  section_names: string[];
  view_count: number[];
  coverable: number[];
  pose_indptr: number[];
  pose_indices: number[];
  lane_xy: number[];
  part_ids?: number[]; // link of each point (absent in trajectories generated before part ids)
  part_names?: string[];
}

export interface TrajectorySummary {
  name: string;
  mode?: string;
  poses?: number;
  coverage?: number;
  generated?: string;
  error?: string;
}
