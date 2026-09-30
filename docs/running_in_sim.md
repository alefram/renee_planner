# Running an experiment in simulation

Every command runs from `renee_simulation/`. The experiments use the
`scanning` compose service: the scanning world, SLAM localization on
`maps/scanning_map` and Nav2. They also need the `moveit` service.

## 1. Start the sim

Run it headless (no Gazebo GUI, no MoveIt RViz), with only the navigation
RViz for watching. The Gazebo GUI plus two RViz windows saturate a 4 GB GPU.

```bash
HEADLESS=true NAV_RVIZ=true docker compose up -d scanning
# wait until Nav2 is active:
docker exec renee_simulation-scanning-1 bash -c \
  'source /fnh_pkgs/install/setup.bash; ros2 lifecycle get /robot/bt_navigator'
HEADLESS=true docker compose up -d --no-deps moveit
# wait for "You can start planning now" in: docker compose logs moveit
```

Check the localization before starting. The robot spawns at
(-0.85, -3.30, yaw 1.57):

```bash
docker exec renee_simulation-scanning-1 bash -c 'source /fnh_pkgs/install/setup.bash; \
  ros2 run tf2_ros tf2_echo robot_map robot_base_footprint' | grep -m2 -E "Translation|RPY"
```

### If the localization is wrong

For example, if someone clicked "2D Pose Estimate" in RViz by accident.
First read the true pose from Gazebo. The map is the world shifted by
(-3, -3):

```bash
docker exec renee_simulation-scanning-1 bash -c \
  'gz topic -e -t /world/table_cube_world/pose/info -n 1' | grep -A12 '^  name: "robot"$'
```

Then send that pose to SLAM. Below is the spawn pose, yaw 90°:

```bash
docker exec renee_simulation-scanning-1 bash -c 'source /fnh_pkgs/install/setup.bash; \
  ros2 topic pub --once -w 1 /initialpose geometry_msgs/msg/PoseWithCovarianceStamped \
  "{header: {frame_id: robot_map}, pose: {pose: {position: {x: -0.85, y: -3.3}, \
  orientation: {z: 0.7071068, w: 0.7071068}}}}"'
```

## 2. Record the corner video (optional)

This works headless:

```bash
docker exec -d renee_simulation-scanning-1 bash -c 'source /fnh_pkgs/install/setup.bash; \
  python3 /fnh_pkgs/src/renee_simulation/renee_trajectory_generation/scripts/record_iso_video.py \
  --no-follow --eye 3.6 2.7 4.8 --target -0.5 -0.5 0.3 --hfov 1.3 \
  --output /tmp/hqp_videos/cleaning_corner.mp4 > /tmp/iso_video.log 2>&1'
```

To stop it cleanly, send SIGINT and wait for "Saved" in `/tmp/iso_video.log`.
Then copy the video into the run folder as `corner_video.mp4`.

## 3. Run the experiment

```bash
docker exec renee_simulation-scanning-1 bash -c 'cd /fnh_pkgs && source install/setup.bash && \
  ros2 launch renee_trajectory_generation trajectory_controller.launch.py experiment:=cleaning_sim'
```

The node logs the planned stops first, for example `Base stop 1: (-2.35,
-1.90, +180 deg) … shift 0.20 m`. It then logs each step as it runs
(`Step 5/13: move_frame …`). The run ends with a results table and
`Trajectory written to …`.

The sim runs at roughly 0.2× real time on this machine, so a full cleaning
run takes about 15 minutes of wall time. The timeouts are in sim time.

### What to watch for

- **Failures in the log**: `unreachable_ik` (it includes the MoveIt IK
  codes), `approach_failed`, `collision`, `timeout`, `localization_jump`,
  or `Traceback`.
- **Localization**: compare the SLAM pose with the Gazebo pose; an error
  over 0.5 m means SLAM has slid.
- **A frozen `/clock`**, or no new step for several minutes.
- **If the robot collides, restart the experiment.** If a Nav2 goal keeps
  running after the node is killed, restart the whole sim.

## 4. After the run

```bash
docker compose stop
```

Outputs, in `data/<experiment>/<YYYYmmdd_HHMMSS>/` (git-ignored):

| File | Contents |
|---|---|
| `summary.txt` | Results table, one line per target |
| `<mission>_run.npz` | Every control tick (tracking errors, base, arm joints, HQP residuals), the plan and the results |
| `<mission>_run.png` | Its figure, drawn afterwards: `python3 scripts/plot_run.py <run folder>` |
| `plan.yaml` | The planned stops and steps |
| `trajectory.yaml` | What was executed: each step's base pose, arm joints and MoveIt goal, plus the pass samples every 2 cm |
| `experiment.yaml`, `node_parameters.yaml` | An exact copy of what ran |
| `node.log`, `corner_video.mp4` | Copied in by hand from the container |
| `captures/`, `captures.yaml` | screw_detection only |

The executed trajectory is also written to
`config/trajectories/<experiment>.yaml`. Each run of that experiment
overwrites it. The container user needs write access there:
`setfacl -m u:usrict:rwx config/trajectories`.
