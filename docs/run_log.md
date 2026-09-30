# Run log

The newest runs are first. The run folders are in `data/<experiment>/`.

## 2026-09-28: cleaning with the arm fixed

- **`20260928_103027`, both passes done.**
  - 45° going out: 2.20/2.20 m, cross-track ≤ 4.2 cm.
  - Perpendicular in reverse: 2.35/2.35 m, cross-track ≤ 2.7 cm.
  - The arm stayed locked in each pass, and the base had zero sideways
    velocity.
- **`20260928_101446`, 45° pass `unreachable_ik`** (MoveIt code -31,
  NO_IK_SOLUTION, on all rolls). The return pass was done.
  - **Cause**: Nav2 left the base at (-2.26, -1.83) instead of the planned
    (-2.35, -1.90). The relative shift then kept that offset, so the base
    was 7.5 cm farther from the rail than planned. The pose is reachable
    from the planned stop: MoveIt returned 11 of 12 rolls.
  - **Fix** (`mission.py`, `move_base`): after Nav2 arrives, the base moves
    straight to the exact planned stop, and the lane anchor and the shift
    start from there.
- **Change**: `arm_fixed` added to cleaning targets (`cleaning.py`,
  `tasks.py`). The arm joints are locked during the pass, and only error
  along the line slows it down.
- Localization reset by hand before the first run: it had been moved in
  RViz by accident (see [running_in_sim.md](running_in_sim.md)).

## 2026-09-26

- **screw_detection `20260926_212847`**: 0.87/0.87 m and 10/10 captures,
  after adding the IMU system plugin to the world.
- **cleaning, several runs**:
  - The return pass (perpendicular, reverse) worked in 3 runs.
  - The 45° pass worked with the old travel posture. It then failed with
    `unreachable_ik` after the travel posture was rotated to face the
    machine. That was fixed on 2026-09-28 (above).
- **Problems found and fixed**:
  - SLAM slid 2.2–2.9 m along the front lane, and the robot hit the wall at
    (-6.18, -1.71). Fixed with the SLAM pause at stops, the jump guard, and
    a ground-truth monitor during runs.
  - An overshoot of 0.65 m past stop 2 caused a whole extra lap and then
    `unreachable_ik`. Fixed with `max_overshoot_m` and backing up straight
    to the stop.
  - A mirrored travel posture was in collision (START_STATE_IN_COLLISION,
    forearm vs camera). Postures are now checked with
    `check_state_validity`.
  - The planner's reach filter ignored the 0.15 m nozzle.
  - The launch file left the capture server running (fixed with `Shutdown`
    on exit). Results were lost on SIGINT (fixed with longer SIGTERM
    timeouts).

## 2026-09-25

- scanning_sim and cleaning_whole_body_sim runs, from before the
  counterclockwise loop and the target refactor. Superseded.
