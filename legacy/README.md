# renee_trajectory_generation (legacy missions)

> **Legacy (frozen).** These are the previous HQP missions, kept so they still
> build and run. Their code is in `renee_trajectory_generation/legacy/`, their
> nodes, launch files, scripts and configs in `legacy/` (`legacy/config/experiments`,
> `legacy/config/trajectories`). The launch names did not change
> (`ros2 launch renee_trajectory_generation trajectory_controller.launch.py ...`).
> Paths below such as `config/experiments/` or `scripts/` now live under `legacy/`.
> The package's current purpose is in the top-level [README](../README.md).

ROS 2 package for the RB-VOGUI+ with UR5e working around a machine (the
Campetella): **missions** (scanning, cleaning, screw detection) that reach a
list of targets with the camera or a tool. Nav2 drives the base between
planned stops (boat mode), MoveIt moves the arm, and a whole-body
Hierarchical Quadratic Programming (HQP) controller refines the poses and
runs the passes along lines (`trajectory_controller_node.py`).

How it works, how to run it in sim, and the status of each mission: [docs/](docs/README.md).

## Components

| Module | Responsibility |
| --- | --- |
| `hqp/robot.py` | Pinocchio whole-body kinematic model (planar base + UR5e arm), state and Jacobians |
| `hqp/tasks.py` | HQP tasks (`joint_limits`, `base_lane`, `camera_view`, `line_path`, `posture`, `base_velocity`) and `build_tasks()` |
| `hqp/solver.py` | The `hqp()` lexicographic QP cascade (OSQP) and the `HQPController` orchestrator |
| `mission.py` | What every mission shares, without ROS: loads the robot and HQP, plans the base stops, runs the generic steps (localize, move_arm, move_base, shift_base, move_frame); `create_mission()` picks the mission from `experiment.mission` |
| `missions/cleaning.py` | Cleaning mission: line path targets for the nozzle, `follow_path` step (HQP pass with the base on its lane) |
| `missions/defect_detection.py` | Defect detection mission (no HQP, no Pinocchio, no venv): per camera pose target a flat list of steps (choose base, stow, back to lane, drive with Nav2, lateral approach, MoveIt reach, record), the footprint clearance checked while driving; camera-pose plot and results |
| `missions/screw_detection.py` | Screw detection mission: the cleaning passes with the camera straight above the row of holes, stopping every `capture.spacing_m` to capture RGB-D keyframes (`/capture_rgbd`) for a later detection step |
| `navigation.py` | Boat-like base navigation without ROS: the loop (`LoopPath`), base placement (`BasePlanner`), localization checks, the base gate |
| `campetella.py` | The Campetella's collision boxes from its published URDF (`moveit.collision_from_description`): box collisions down its fixed-joint tree, placed in `targets_frame` |
| `arm_motion.py` | MoveIt clients for the arm: IK, state validity, planning-scene collision objects, joint motions |
| `trajectory_controller_node.py` | ROS 2 node, interface only: reads `joint_states`/`odom`/TF, sends Nav2 and MoveIt requests, publishes `cmd_vel`/`joint_trajectory` |
| `recorders.py` | Run recorders of the camera pose (defect detection) and path (cleaning, screw detection) missions: per-tick data, plan and results saved as `<mission>_run.npz`, and the `summary.txt` tables |
| `scripts/plot_run.py` | Plots a run from its `<mission>_run.npz`: `python3 scripts/plot_run.py data/<experiment>/<stamp>` |
| `arm_static_planner_node.py` + `launch/arm_static_planner.launch.py` | Arm-only camera views, base fixed (no Nav2, no HQP): for each target frame (TF, or a pose in the YAML) MoveIt places the camera `view.standoff_m` along the target's +Z looking at its origin; results in `results.yaml`. `ros2 launch renee_trajectory_generation arm_static_planner.launch.py experiment:=arm_static_planner_sim`. The goal math is in `arm_static_planner.py`; `launch/arm_static_planner_frames.launch.py experiment:=<name>` publishes the targets and tool goals as TF frames (for RViz) without moving the robot |
| `launch/trajectory_controller.launch.py` | Runs the node with one experiment YAML (and the `/capture_rgbd` server when the YAML asks for it) |
| `scripts/record_iso_video.py` | Records a video of the Gazebo run from a fixed or following camera |

Each experiment is one YAML in `config/experiments/`. It holds everything for
the run: the priority levels and tasks solved each control cycle, the
targets, MoveIt and navigation settings, and under
`experiment.node_parameters` the node's runtime parameters (sim or real robot
topics, control rate, OSQP tolerances).

## Dependencies

Build and run the package inside the ROS Jazzy container. Install the numerical
dependencies in a virtual environment with access to the system ROS packages:

```bash
source /opt/ros/jazzy/setup.bash
python3 -m venv --system-site-packages /tmp/renee-hqp-venv
source /tmp/renee-hqp-venv/bin/activate
python -m pip install -r src/renee_simulation/renee_trajectory_generation/requirements-hqp.txt
```

The ROS image also ships its own Pinocchio (built for NumPy 1.x), which is
found before the venv's. Put the venv's Pinocchio and native libraries first
after activating the venv:

```bash
C=/tmp/renee-hqp-venv/lib/python3.12/site-packages/cmeel.prefix
export PYTHONPATH=$C/lib/python3.12/site-packages:/tmp/renee-hqp-venv/lib/python3.12/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=$C/lib:$LD_LIBRARY_PATH
```

If `python3 -m venv` fails because `ensurepip` is missing, create it with
`python3 -m venv --without-pip --system-site-packages /tmp/renee-hqp-venv`;
the system `pip` is used instead.

The version bounds in `requirements-hqp.txt` keep Pinocchio, Coal, urdfdom and
TinyXML binary-compatible with the Python 3.12 environment used by Jazzy.

## Build

From the workspace root:

```bash
source /opt/ros/jazzy/setup.bash
source /tmp/renee-hqp-venv/bin/activate
colcon build --packages-select renee_trajectory_generation --cmake-args -DBUILD_TESTING=OFF
source install/setup.bash
```

## Usage

Run an experiment by name (`config/experiments/<name>.yaml`) or by path,
with Nav2 and move_group already running:

```bash
ros2 launch renee_trajectory_generation trajectory_controller.launch.py experiment:=defect_detection_sim
```

An unknown name lists the experiments. With a symlink install the YAML is
read from the source tree, so editing it needs no rebuild. The outputs go to
`data/<experiment name>/<YYYYmmdd_HHMMSS>/` (git-ignored): `experiment.yaml`
and `node_parameters.yaml` (exact copy of what ran), `plan.yaml` (the base
stops and steps), the mission's plot and `.npz`, `summary.txt` (the results
table), `trajectory.yaml` and, for screw_detection, `captures/` (the RGB-D
keyframes of `/capture_rgbd`) and `captures.yaml`.

The executed trajectory is also kept with the configs, in
`config/trajectories/<experiment name>.yaml` (overwritten by each run of that
experiment), all in `robot_map`: the planned stops, where each step left the
robot (base pose, arm joints, the MoveIt goal of each target), and each
pass sampled every 2 cm (path parameter, base pose, arm joints).

| Experiment | Mission |
|---|---|
| `cleaning_sim` | two nozzle passes along the transversal rail with the arm fixed: 45° from above going, perpendicular to the front face coming back with the base in reverse |
| `defect_detection_sim` | 13 camera poses around the Campetella (authored in web_tf_editor): Nav2 drives the base on a loop generated from the machine's boxes (>= 0.30 m clearance), MoveIt places the camera; no capture yet |
| `motion_test_real` | real robot, real map: base around the Campetella's CAD reference, pointer_tester tip at 3 rail points ("Real robot: motion test") |
| `screw_detection_sim` | slow camera pass straight above the rail's row of holes (between the front supports), stopping every 0.1 m to capture RGB-D keyframes |

Every experiment YAML has the same numbered sections, in this order (a
section an experiment doesn't need keeps its header with "Not used in this
experiment"):

| # | Section | Keys |
|---|---|---|
| 1 | EXPERIMENT | `experiment` (name, `mission`, description, `node_parameters`) |
| 2 | ROBOT AND FRAMES | `wrist_camera`, `ur_type`, `nozzle`, `targets_frame`, `frames` |
| 3 | MOVEIT | `moveit` (`collision_from_description`: the Campetella's boxes from its model), `travel_arm_q` |
| 4 | NAVIGATION | `navigation` (workspace, localization checks, `loop_path`) |
| 5 | TARGETS | `base_placement`, `capture` (screw_detection), `targets` (frame poses or line paths; base stops are planned) |
| 6 | HQP TASKS | `levels` |

To add an experiment, copy one and change `experiment.name` and whatever the
experiment tests. Sim or real is set per YAML in `experiment.node_parameters`.
The real robot needs these values (the base twist enters the robot's
`twist_mux` through `vogui_ros1_ros2_bridge`, so the e-stop and teleop still
override it; the arm runs with ur_robot_driver, `tf_prefix: robot_arm_`):

```yaml
experiment:
  node_parameters:
    use_sim_time: false
    cmd_vel_topic: /robot/docker/cmd_vel      # must be relayed by the bridge
    cmd_vel_stamped: false
    joint_trajectory_topic: /robot/scaled_joint_trajectory_controller/joint_trajectory
```

Coordinates (loop, targets, collision objects) are those of the
map in use: a real-robot experiment needs the real map's values.

## Real robot: motion test

`motion_test_real` tests the whole robot on the real map: Nav2 drives the base
on the loop around the Campetella and MoveIt aims the pointer_tester tip at
three points of its front rail (`defect_detection` mission, no HQP, so no
venv needed). There is no Campetella in the lab, so its CAD is published as a
reference only, at a pose you choose on a free area of the map. The targets
are written relative to `campetella_base_link` (`targets_origin`) and follow it.

1. Base: `docker compose up bridge-real localize_real navigation-real`.
2. Arm with the pointer (External Control running on the pendant):

   ```bash
   ros2 launch renee_action_servers bringup_actions.launch.py real_robot:=true \
     robot_ip:=192.168.0.101 reverse_ip:=192.168.0.150 use_rviz:=true
   ```

3. The Campetella reference: `campetella_base_link` at (x, y, yaw) in
   `robot_map`, z 0.8 (its CAD origin, as in sim). Pick a free area with room
   for the loop (the machine plus ~1.5 m on every side) and check it in RViz
   (add a RobotModel on `/campetella_robot_description`):

   ```bash
   ros2 launch campetella_sim spawn_campetella.launch.py gazebo:=false \
     parent_frame:=robot_map use_sim_time:=false x:=1.0 y:=0.0 z:=0.8 yaw:=0.0
   ```

4. Run (slow arm: `velocity_scaling: 0.1`; e-stop at hand):

   ```bash
   ros2 launch renee_trajectory_generation trajectory_controller.launch.py experiment:=motion_test_real
   ```

To give another target, add a line to `targets` in
`config/experiments/motion_test_real.yaml`, in `campetella_base_link`: the
pointer tip at `position`, its +Z aimed at `look_at` (0.3-1.5 m apart), e.g.

```yaml
  - {name: my_point, frame: pointer, position: [-1.2, 0.45, 0.3], look_at: [-1.2, -0.25, 0.0]}
```

Without `targets_origin` the targets are in `robot_map` directly.

## Targets, base stops and steps

An experiment lists **targets** (section 5), all in `targets_frame`:

```yaml
frames:
  camera: robot_arm_rgbd_camera_left_camera_optical_frame
  nozzle: nozzle_tip
targets:
  # A frame pose: the frame's origin at `position`, its +Z aimed at `look_at`.
  - {name: rear_2_1, frame: camera, position: [-3, -4.5, 1], look_at: [-3, -3.2, 0.7]}
  # A path: the frame (line_path task) follows the line while the base drives along its lane.
  - {name: transversal_top, frame: nozzle, path: {start: [-4.6, -3.02, 0.87], end: [-2.5, -3.02, 0.87]}}
```

The camera pose follows the `camera_view` task: optical axis (+Z) straight
at `look_at`, image upright with `roll: 0.0` (6 DoF) or roll free with
`roll: null` (5 DoF); `position`-to-`look_at` must be within
`view_distance_m`. A pose counts as reached when the position and aim errors
stay below `reach_tolerance_m` / `reach_tolerance_rad` for 1 s. `refine:
false` on a target scores it once the arm is static after MoveIt, without
the HQP.

**No base poses are written.** At startup `navigation.BasePlanner` places the
base stops on `navigation.loop_path` (boat mode: the base on the loop, its
heading along it, driving forward only). For each target it samples the
loop every `base_placement.spacing_m`, solves the arm IK locally (base
fixed) and keeps the spots where the frame reaches the pose within the joint
limits, with the arm clear of the `moveit.collision_objects` boxes. A
straight sideways shift towards the machine (`base_placement.offsets_m`) is
tried only for targets no spot on the loop reaches. Then it groups the
targets into the fewest stops along the driving direction, from
`navigation.start_pose`, so the base drives the loop once. The stops are
logged and written to `plan.yaml`.

The stops become a list of **steps**, run one after the other by the
mission (`experiment.mission`: `cleaning`, `screw_detection` or `defect_detection`).
The generic steps live in `mission.py`, each mission adds its own per target
(`follow_path` in `missions/cleaning.py`, reused by `missions/screw_detection.py`).
`defect_detection` does not use them: it is its own linear state machine
(`missions/defect_detection.py`) over `navigation.py`'s rules:

| Step | Executed by | What it does |
|---|---|---|
| `localize` | - | waits for stable localization at `navigation.start_pose` |
| `move_arm` | MoveIt | joint motion (Pilz PTP, retried with `fallback_pipeline`), e.g. to `travel_arm_q` before driving |
| `move_base` | Nav2 | `FollowPath` along the loop to the stop, then the base is brought aligned with the lane and static |
| `shift_base` | node | straight sideways shift (only if the stop needs it), aligned and static; undone before leaving |
| `move_frame` | MoveIt | `compute_ik` (seeded with the planner's solution) and the joint motion to the target's frame pose |
| `refine_pose` (scanning) | HQP | refines the camera pose (`camera_view`, base held), with `check_state_validity` 0.25 s ahead at `validity_rate_hz` |
| `follow_path` (cleaning, screw_detection) | HQP | the pass: `line_path` keeps the frame on the line, `base_lane` drives the base forward on its lane |

Nav2 and MoveIt execute the large motions; the HQP only refines poses and
runs path passes. A failed step marks its targets (`nav_failed`,
`unreachable_ik`, `approach_failed`, `stow_failed`, `collision`, `timeout`)
and skips the rest of its target or stop; stowing and shifting back onto the
lane still run.

The node needs move_group (the `moveit` compose service, started with
`--no-deps` next to `scanning`) and Nav2. For faster runs, start both with
`HEADLESS=true` (Gazebo server only, no RViz):

```bash
HEADLESS=true docker compose up scanning
HEADLESS=true docker compose up --no-deps moveit
```

At the end (and on Ctrl+C) the run folder (`output_dir/<YYYYmmdd_HHMMSS>/`)
gets `<mission>_run.npz` (every control tick, the plan and the results),
`summary.txt`, `plan.yaml` and copies of the YAML and the node parameters.
The figures are drawn afterwards from the `.npz`:

```bash
python3 scripts/plot_run.py data/<experiment>/<YYYYmmdd_HHMMSS>   # writes <mission>_run.png
```

While the HQP streams arm commands, the node integrates its own arm reference
instead of resetting the model to the measured joints every cycle: a
position-controlled arm tracks with a lag, and commanding "measured + v·dt"
made the arm (and the cleaning pass) stall. The reference is resynced to the
measurement when they differ by more than `arm_reference_reset_rad` (0.15 rad)
and whenever the HQP is not streaming.

## HQP behavior

The whole-body decision variable is a 9-dimensional velocity vector: the
mobile base's planar twist (`vx, vy, wz`, base frame) plus the six UR5e joint
velocities. `solver.hqp(levels, nv)` solves the configured tasks as a
lexicographic cascade of QPs (OSQP): at each priority level it minimizes that
level's weighted least-squares residual, subject to (a) every inequality
carried over from previous and current levels, and (b) the exact task value
already achieved by every higher-priority level, frozen as an equality
constraint. `tasks.build_tasks(config, robot)` turns the experiment YAML into the
task objects consumed by each level.

The mission runs it online through `trajectory_controller_node.py`, which
subscribes to `/robot/joint_states` and `/robot/robotnik_base_control/odom` and
publishes to `/robot/robotnik_base_control/cmd_vel` and
`/robot/joint_trajectory_controller/joint_trajectory` at a fixed control rate.
