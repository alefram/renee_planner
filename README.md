# renee_trajectory_generation

Generates the **scan trajectory of the Campetella**: the ordered list of
camera poses (the ZED2i left optical frame, in `robot_map`) that covers the
machine's surface, one forward lap around it. The package only **generates**
the poses. It never moves the robot. A Behavior Tree executes them later:
Nav2 drives the base and MoveIt2 places the camera (reference tree:
`behavior_trees/scan_offline.xml`).

There are two modes:

- **cad**: from the Campetella's CAD (the STL meshes of its URDF).
- **sensor**: from a previous RGB-D scan (a `/capture_rgbd` session). The scan
  is aligned to the CAD with ICP, which also corrects the hand-measured
  machine pose, and completed with the CAD where the scan did not reach.

A web viewer shows the results: the machine, the camera frustums, the route
and the coverage.

`hqp/` holds the whole-body HQP model. `legacy/` holds the previous missions,
frozen; see [legacy/README.md](legacy/README.md). The new code does not import
either of them.

## Structure

```
renee_trajectory_generation/      Python package, no rclpy
  machine.py  surface/{base,cad_mesh,sensor_tsdf}.py  camera.py  workspace.py
  candidates.py  visibility.py  set_cover.py  ordering.py  trajectory.py  pipeline.py
  hqp/  legacy/
interfaces/msg/{CameraPose,CameraTrajectory}.msg   interfaces/action/GenerateScanPoses.action
src/scan_pose_generator_node.py   src/scan_viewer_server_node.py
launch/scan_pose_generator.launch.py   launch/scan_viewer.launch.py
web/                              viewer: React + TypeScript + Vite + Tailwind + react-three-fiber (dist/ committed)
behavior_trees/scan_offline.xml   reference tree for the executor
config/experiments/campetella_scan_{cad,sensor}_{sim,real}.yaml
config/trajectories/              output: <experiment>.yaml, _coverage.npz, _machine.glb
legacy/                           previous nodes, launch files, scripts and configs
```

The pipeline runs **machine → surface → target → workspace → candidates → visibility → set_cover → ordering → trajectory**:

| File | What it does | Input → Output |
|---|---|---|
| `machine.py` | What the machine is and where it is | URDF XML, `campetella_config.yaml`, robot_map → campetella_base_link → mesh parts (STL, scale, pose), collision boxes, `T_map_machine` |
| `surface/base.py` | Common surface format | → `SurfaceModel`: points, normals, section per point, occluder mesh, `raycast()` |
| `surface/cad_mesh.py` | CAD mode | Machine → STLs merged and decimated (quadric), Poisson-disk samples (~2.5 cm) with normals; bottom faces and contact faces dropped; viewer `.glb` |
| `surface/sensor_tsdf.py` | Sensor mode | `frames.jsonl` (depth, intrinsics, `T_world_camera`) → TSDF → marching cubes → samples; ICP point-to-plane against the CAD → corrected machine pose; CAD completion |
| `target.py` | What to inspect | surface + `target:` (sections, parts by link name with wildcards, exclusions) → the surface subset; the rest of the machine still occludes and still defines the lane |
| `camera.py` | Camera model | position + look_at → optical-frame quaternion (+Z forward, upright image); `sees()`: frustum (60×40°), range (0.3–1.5 m), incidence (< 60°) |
| `workspace.py` | Where the camera can be (no IK) | machine boxes, robot width, clearance, arm reach → base lane (loop around the machine), `inside()`, `arc_length()` |
| `candidates.py` | Possible poses | surface patches (voxel × normal direction) × standoffs × tilts → reachable poses with a clear view of their patch |
| `visibility.py` | What each candidate sees | candidates, surface, camera → sparse matrix candidate × point, with occlusion (open3d raycasting) |
| `set_cover.py` | Fewest poses | visibility, costs → greedy weighted set cover up to the target coverage (95%) |
| `ordering.py` | Visiting order | poses, lane → sorted by arc length (one forward lap) + 2-opt within a 0.6 m window |
| `trajectory.py` | Result I/O | poses + coverage → `config/trajectories/<exp>.yaml` + `_coverage.npz` |
| `pipeline.py` | Runs everything | experiment YAML + URDF + machine pose → Trajectory; caches surface, candidates and visibility by config hash |

| Node | Input | Output |
|---|---|---|
| `scan_pose_generator_node.py` | experiment YAML, `/campetella_robot_description`, TF `robot_map → campetella_base_link`, goal on `/generate_scan_poses` | trajectory files, `/scan_trajectory` (latched `CameraTrajectory`), action result |
| `scan_viewer_server_node.py` | `web/dist`, `config/trajectories/` | viewer on `http://<host>:8091`. It picks up new or regenerated trajectories on its own (polls every 3 s, no ROS needed); rosbridge, if installed, adds the camera's current pose from `/tf` |

## Fixed ring (`poses.method: ring`)

Instead of the fewest poses that cover the surface, a fixed lap: one pose
every `ring.spacing_m` along the lane. The camera is `ring.inset_m` towards
the machine, at `ring.height_m` (null: the machine's top + `above_top_m`,
capped at `workspace.camera_height_m`), and looks at the machine `ring.pitch_deg`
down from horizontal. Every pose is kept. The surface and visibility are
still computed, so the viewer shows what the ring sees and what it misses
(red). `ring.extra_rows` add poses at the same lane points with their own
height and pitch (negative: looking up); a row with `sections` only keeps the
poses within `near_m` of them (e.g. the tall vertical column). Example:
`config/experiments/mapping.yaml`.

| File | What it does | Input → Output |
|---|---|---|
| `ring.py` | Fixed ring | lane, machine boxes, `ring:` → poses in lap order (`Candidates`) |

## Inspecting only some parts

Set the experiment's `target:` section. Use one experiment YAML per target, so
each one gets its own trajectory file.

```yaml
target:
  sections: [transversal]          # extraction, transversal, vertical, wrist, cart
  parts: [vertical27, wrist*]      # URDF link names, with or without _link; shell wildcards
  exclude_parts: [cart*]
```

A surface point is kept if its section **or** its part is listed. Everything
empty means the whole machine. If nothing matches, the error lists the
available sections and parts. Only the surface to cover changes: the other
parts still block the camera's view, and the lane still goes around the whole
machine.

## Output

`config/trajectories/<experiment>.yaml`. This is what the executor reads.

```yaml
experiment: campetella_scan_cad_sim
mode: cad
frame: robot_map
camera: {frame: robot_arm_rgbd_camera_left_camera_optical_frame, fov_deg: [60, 40], view_distance_m: [0.3, 1.5], ...}
coverage: {ratio: 0.93, coverable_ratio: 0.95, surface_points: ..., uncovered_points: ..., file: <exp>_coverage.npz, ...}
machine: {root_link: campetella_base_link, position: [...], orientation: [...], corrected_by_icp: false, mesh: <exp>_machine.glb}
poses:   # execution order
  - {id: 0, name: vertical_000, section: vertical, position: [x, y, z], orientation: [qx, qy, qz, qw],
     look_at: [x, y, z], covered: 812, arc_m: 0.42, standoff_m: 0.8, tilt_deg: 0.0}
```


**Robot model:** the generator only uses the geometric workspace (the lane +
arm reach + camera height + clearance from the machine). The executor decides
with MoveIt (IK, joint limits, collisions) whether the robot can reach each
pose, and skips it if not (`RecordSkippedPose` in the BT).

## Dependencies

The generator needs `open3d`, `numpy`, `scipy` and `PyYAML` (`requirements.txt`).
`scipy` and `PyYAML` come from apt in the ROS image. Inside the sim container,
install `open3d` directly. Keep NumPy < 2, because the ROS packages were built
against apt's NumPy 1.26. This lasts until the container is recreated.

```bash
pip install --break-system-packages "open3d>=0.19" "numpy<2"
```

Or use a venv that can see the ROS packages:

```bash
source /opt/ros/jazzy/setup.bash
python3 -m venv --system-site-packages /tmp/renee-scan-venv
source /tmp/renee-scan-venv/bin/activate
python -m pip install -r src/renee_simulation/renee_trajectory_generation/requirements.txt
```

The sim image has no `ensurepip`, so `python3 -m venv` fails there. Use
`python3 -m venv --without-pip --system-site-packages /tmp/renee-scan-venv`
instead. The venv's `python -m pip` is then the system pip, and it installs
into the venv. `/tmp` is a volume shared by the sim containers, so the venv
outlives them. Activate it in every shell before launching the generator.

The viewer needs `rosbridge_server` at runtime. Node/npm are only needed to
rebuild `web/dist`:

```bash
cd web && npm install && npm run build      # npm run dev: hot reload, proxies /api to port 8091
```

## Build

```bash
colcon build --symlink-install --packages-select renee_trajectory_generation
source install/setup.bash
```

This also generates the msgs and the action. With `--symlink-install`, the
experiment YAMLs are read from the source tree, the results go to the
source's `config/trajectories/`, and the cache goes to `data/scan_cache/`
(git-ignored). Without it, everything goes to `~/.ros/scan_poses/`.

## Usage

```bash
# Generator (waits for goals; run_on_start:=true generates once at startup)
ros2 launch renee_trajectory_generation scan_pose_generator.launch.py experiment:=campetella_scan_cad_sim
ros2 action send_goal --feedback /generate_scan_poses \
  renee_trajectory_generation/action/GenerateScanPoses "{use_cache: true}"
# Another experiment or a capture session, from the goal:
#   "{experiment: campetella_scan_sensor_sim, captures_dir: /tmp/renee_scan_session, use_cache: true}"

# Viewer
ros2 launch renee_trajectory_generation scan_viewer.launch.py     # http://localhost:8091
```

**Sim vs real:** only the experiment YAML changes (`use_sim_time`, the
machine pose source, the captures folder). The generator needs the machine's
description and its TF. In sim, the Campetella spawn provides both. On the
real robot, use `spawn_campetella.launch.py gazebo:=false parent_frame:=robot_map
use_sim_time:=false x:=... y:=... z:=0.8 yaw:=...` at the measured pose. You
can also set `machine.urdf_file` and `machine.pose_source: yaml`.

If the generator runs where the URDF's `file://` mesh paths do not exist
(another container), map them with `machine.mesh_path_map`.

## How to test it

1. **Build**: `colcon build ...`. Check that `ros2 interface show renee_trajectory_generation/action/GenerateScanPoses` works and that the legacy launch files still find their YAMLs.
2. **CAD mode in sim**:
   - Start `scanning_entrypoint.sh` (headless).
   - Launch the generator with `experiment:=campetella_scan_cad_sim` and send the goal.
   - The log prints each stage, then the pose count and the coverage (target 95% of the reachable points).
   - A second goal with `use_cache: true` should only redo set cover and ordering.
3. **Viewer**:
   - Launch `scan_viewer.launch.py` and open `http://localhost:8091`.
   - Check that the frustums are outside the machine and within reach, that the route makes one forward lap (blue → red), and that the coverage has no big red (not covered) areas.
   - Click a pose: the points it sees turn yellow.
4. **CAD mode on the real robot (no motion)**:
   - Publish the Campetella reference with `gazebo:=false parent_frame:=robot_map`.
   - Run `campetella_scan_cad_real`.
   - Check in the viewer that the poses sit on free map areas. With rosbridge on, the live camera frustum (magenta) shows where the real camera is.
5. **Sensor mode**:
   - Use a `/capture_rgbd` session from an earlier run (sim first).
   - Generate `campetella_scan_sensor_sim` and compare it with the CAD result ("Compare with").
   - To test ICP, set `machine.pose_source: yaml` with `machine.pose.xyz` a few cm off the true pose ((-2, -3, 0.8) in sim). The generated `machine.position` should come back close to the true pose (`corrected_by_icp: true`).
