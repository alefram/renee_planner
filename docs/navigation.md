# Base navigation

All coordinates are in `robot_map` (the scanning world's SLAM map). The
Campetella is centred around (-3.3, -3.1). Its long transversal rail spans
x -4.85 → -1.68, y -3.27 → -2.90, with its top at z ≈ 0.87.

## The loop (boat mode)

The base drives a fixed closed loop (`navigation.loop_path`) with Nav2
`FollowPath`. It always drives forward with its heading along the path, and
it never turns in place.

- Corners: (-0.85, -1.7) → (-5.85, -1.7) → (-5.85, -5.0) → (-0.85, -5.0),
  with arcs of radius 0.4 m.
- **Counterclockwise only**, with the machine on the robot's left. This is a
  constraint of the real robot's cabling.
- **Front lane**, y = -1.70, driven toward -X. The cleaning and screw
  detection passes run here.
- **Start**, (-0.85, -3.3) facing +Y (yaw 90°). The spawn pose, the SLAM
  `map_start_pose` and `navigation.start_pose` all use it.
- The 1.2 × 0.7 m footprint keeps ≥ 0.29 m from the machine, its supports
  and the walls along the whole loop.

## Base stops (automatic placement)

No base poses are written in the YAMLs. At startup `BasePlanner` works them
out:

1. It samples the loop every `base_placement.spacing_m` (0.05 m).
2. For each spot, it solves the arm IK with the base fixed, then checks the
   joint limits and reach (including the 0.15 m nozzle), and that the arm
   and camera stay clear of the `moveit.collision_objects` boxes.
3. It only tries a straight sideways shift toward the machine
   (`base_placement.offsets_m`: 0.1, 0.2 and 0.3 m) if no spot on the loop
   works.
   - The footprint must stay clear at the stop and at the end of the pass.
   - For a reverse pass, the end is behind the stop.
4. It groups the targets into the fewest stops, in driving order, and logs
   the rejection counts: footprint, out of reach, no IK, arm collision.

## At a stop

1. **`move_base`**: Nav2 drives along the loop to the stop. Nav2 stops
   within its goal tolerance, typically a few cm to 10 cm short. The node
   then covers the rest in a straight line, heading held, so the base ends
   exactly at the planned stop.
   - This matters: 7.5 cm farther from the rail was enough to make the 45°
     cleaning pose unreachable (see [run_log.md](run_log.md)).
2. **`shift_base`**: a straight sideways move toward the machine, measured
   from the planned stop, and undone before leaving.
3. **Overshoot**: a pass can leave the base past the next stop. If it is
   less than `max_overshoot_m` (1.0 m) past, the base backs up straight to
   the stop without rotating. Otherwise it would need a whole extra lap.

## Localization safeguards

On the front lane, the lidar sees little more than two parallel walls,
because the rail is above the scan plane. When the robot moves slowly there,
SLAM's scan matching once slid 2.2–2.9 m along the lane and drove the robot
into a wall. The safeguards against this:

- **SLAM pause** (`pause_localization_at_stops: true`): new measurements are
  paused, through `/slam_toolbox/pause_new_measurements`, during all the
  work at a stop. They are resumed only for the Nav2 drives. Odometry alone
  is accurate enough over a stop's work.
- **Jump guard**: if the localization correction jumps more than
  `jump_tolerance` ([1.5 m, 0.5 rad]), the run ends with status
  `localization_jump`.
- **Workspace and settle checks**: no Nav2 goal is sent while the base is
  outside `workspace`, or before the correction is stable for `settle_s`.
