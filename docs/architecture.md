# Architecture

## Who moves the robot

| Motion | Executed by | Why |
|---|---|---|
| Driving between stops | Nav2 `FollowPath` along a fixed loop | Safety on the real robot: Nav2 keeps its costmaps and collision checks |
| Arm travel, stow, approach to a target | MoveIt (Pilz PTP, OMPL fallback) | Collision-checked joint motions against the planning scene |
| Short straight base moves at a stop (sideways shift, final approach to the stop, backing up to a stop) | The node, as a slow gate (≤ `base_lane.max_speed`) | Nav2 drives forward only; these moves are a few cm to ~0.5 m |
| Refining a camera pose, running a pass along a line | Whole-body HQP (base + arm) | Only at a stop, starting from where MoveIt left the arm |

The HQP never drives the robot between stops. During a pass it moves the base
only along its lane: no sideways velocity and no rotation (boat mode).

## Modules (`renee_trajectory_generation/`)

The ROS node is interface only. All the logic is in plain Python modules
with no ROS imports, driven by the node every control cycle.

| Module | Role |
|---|---|
| `mission.py` | Base class `Mission`. Loads the robot model and the HQP levels, plans the base stops, and runs the step list: `localize`, `move_arm`, `move_base`, `shift_base`, `move_frame`, plus each mission's own step. It also holds the localization guard, the SLAM pause, IK retries and the trajectory recording. `create_mission()` picks the class from `experiment.mission` |
| `missions/cleaning.py` | `CleaningMission`: line path targets, `follow_path` step (HQP pass), options `tilt_deg`, `standoff_m`, `reverse`, `arm_fixed` |
| `missions/screw_detection.py` | `ScrewDetectionMission(CleaningMission)`: the same pass with the camera, pausing for `/capture_rgbd` |
| `missions/defect_detection.py` | `DefectDetectionMission` (not a `Mission`: no HQP, no Pinocchio): per camera pose target, choose base (`navigation.base_candidates` + MoveIt IK), stow, back to lane, drive (`LoopPath.boat_path`), lateral approach, reach (MoveIt), record |
| `navigation.py` | `LoopPath` (the boat-mode loop), `BasePlanner` (base stop placement), localization checks, the gate twist |
| `tasks.py` | HQP tasks: `joint_limits`, `base_lane`, `camera_view`, `line_path`, `posture`, `base_velocity` |
| `solver.py` | Lexicographic QP cascade (OSQP) and `HQPController` |
| `robot.py` | Pinocchio model (planar base + UR5e + nozzle and camera frames) |
| `campetella.py` | The Campetella's collision boxes, read from its published URDF at startup (`moveit.collision_from_description`: `/campetella_robot_description` + the TF of `campetella_base_link`), for the planning scene and the base planner |
| `arm_motion.py` | MoveIt clients: `compute_ik`, `check_state_validity`, planning-scene boxes, joint motions |
| `recorders.py` | Run recorders: per-tick data, plan and results saved as `<mission>_run.npz` (plotted afterwards by `scripts/plot_run.py`) |

Other files:

- `src/trajectory_controller_node.py`: the ROS node. It subscribes to
  `joint_states` and odometry, looks up TF, sends the Nav2, MoveIt and
  capture requests, and publishes `cmd_vel` and the arm `joint_trajectory`.
- `launch/trajectory_controller.launch.py`: runs the node for one experiment.
  It also starts the capture server when the YAML asks for it, and stops
  everything cleanly when the node exits.
- `scripts/record_iso_video.py`: records the corner video from a fixed
  Gazebo camera.

## One YAML per experiment

Each experiment lives in a single file in `config/experiments/`, with no
layered configs. The file includes the node's runtime parameters
(`experiment.node_parameters`), so switching between sim and the real robot
is done in the YAML. Its sections, in order:

1. experiment
2. robot and frames
3. MoveIt
4. navigation
5. targets
6. HQP levels

## HQP levels (cleaning, as configured)

1. `hard_limits`: `joint_limits` on the 6 arm joints, and `base_lane`, which
   keeps the base locked, held at a point, or driving its lane.
   - With `arm_fixed`, `joint_limits` also locks the arm (dq = 0).
2. `nozzle_tracking`: `line_path`. The nozzle stays at the standoff from the
   line, aimed at it, and advances along the line at `speed`.
3. `posture_and_base_regularization`: `posture` pulls the arm toward its
   configuration at the start of the pass, and `base_velocity` is a small
   regularization.

## Sim vs real robot

| | Sim (Gazebo) | Real |
|---|---|---|
| `use_sim_time` | true | false |
| Base command | `/robot/robotnik_base_control/cmd_vel` (TwistStamped) | `/robot/docker/cmd_vel` (Twist, through `twist_mux` via `vogui_ros1_ros2_bridge`, so the e-stop and teleop still override it) |
| Arm command | `/robot/joint_trajectory_controller/joint_trajectory` | `/robot/scaled_joint_trajectory_controller/joint_trajectory` (ur_robot_driver, `tf_prefix: robot_arm_`) |
| Map coordinates | `maps/scanning_map` | Real map: loop and targets must be re-measured |
| Campetella collision boxes | `/campetella_robot_description` and `world -> campetella_base_link` from `spawn_campetella.launch.py` | The same model must be published, with `campetella_base_link` placed in `robot_map` from the machine's measured pose |

Real-robot constraints already built into the sim setup:

- **The base only turns counterclockwise.** The real robot's cabling can't
  take the other direction. The loop runs counterclockwise with the machine
  on the robot's left, and the robot spawns facing +Y so its first turn is to
  the left.
- **Nav2 and MoveIt stay in charge of the large motions** (the table above).
