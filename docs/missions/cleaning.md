# Cleaning mission (`cleaning_sim`)

Non-contact cleaning of the Campetella's **long transversal rail** with a
virtual nozzle. The nozzle tip is 0.15 m out from the arm flange along
tool0's +Z, which is the spray axis. The rail is cleaned in two passes from
the front lane, one going out and one coming back. **In each pass the arm
holds one fixed configuration**, and only the base moves the nozzle along the
rail.

## The motion

| | Pass 1: `rail_top_45` | Pass 2: `rail_front_face_back` |
|---|---|---|
| Nozzle | 45° from above, aimed at the top face's front edge (y -2.95, z 0.87) | Horizontal, perpendicular to the front vertical face (y -2.90, z 0.735) |
| Distance | 0.60 m | 0.80 m |
| Along the rail | x -2.60 → -4.80 (2.20 m) | x -4.80 → -2.45 (2.35 m) |
| Base | Forward, toward -X, 0.20 m closer than the lane (y -1.90) | **In reverse**, toward +X, on the lane (y -1.70), with no rotation |
| Arm | Fixed (`arm_fixed: true`) | Fixed (`arm_fixed: true`) |

Step by step:

1. Localize at the start. The arm is in the travel posture: folded over the
   base, with the shoulder turned so the wrist and camera face the machine.
2. Nav2 drives to stop 1 (-2.35, -1.70). The base then moves straight to the
   exact stop and shifts 0.20 m sideways toward the rail.
3. MoveIt brings the nozzle to the 45° pose at the start of the line.
4. **Pass 1**: the arm joints are locked. The base drives forward along the
   lane with no sideways speed and no rotation, and the nozzle follows the
   rail.
5. The arm is stowed, and the base shifts back onto the lane.
6. The pass ended past stop 2 (-4.10, -1.70), so the base **backs up** to it
   in a straight line without rotating.
7. MoveIt brings the nozzle to the perpendicular pose.
8. **Pass 2**: the arm is locked, and the base drives **in reverse** along
   the lane.
9. The arm is stowed.

## Why these numbers

- **0.60 m and the front edge, not 0.80 m and the centerline.** At 45° from
  above, the arm can't reach the top face's centerline (y -3.085) from the
  front lane at any distance between 0.5 and 0.8 m. So the pass aims at the
  front edge from 0.60 m.
- **Both passes stop before x ≈ -2.2.** The vertical column
  (x -2.22 → -1.85) stands between the lane and the rail at its +X end.
  The stretch x -2.45 → -1.68 isn't covered from the front lane.
- **Only a 0.20 m shift.** The base footprint must stay clear of the front
  supports at the stop and at the end of the pass.

## How `arm_fixed` works

- The `joint_limits` task locks the arm joints (dq = 0) during the pass.
- `line_path` still sets the pace along the line, but now only the base's
  forward speed can follow it.
- The pass only slows down for error *along* the line (the nozzle falling
  behind). Sideways error can't be corrected with the arm locked, so it is
  only measured, as cross-track.
- The resulting drift comes from the base heading. A heading error below
  `base_lane.align_tolerance_rad` (0.02 rad) is not corrected: 1° over 2.2 m
  gives about 4 cm. Lowering that tolerance would reduce it.

## Latest result (2026-09-28, `data/cleaning_sim/20260928_103027`)

| Pass | Covered | Max cross-track | Standoff | Max aim error | Max vy |
|---|---|---|---|---|---|
| 45° going out | 2.20 / 2.20 m | 4.2 cm | 0.57–0.60 m | 0.060 rad | 0 |
| Perpendicular, reverse | 2.35 / 2.35 m | 2.7 cm | 0.79–0.83 m | 0.024 rad | 0 |

## Open points

- The rail's +X end (x -2.45 → -1.68), behind the column, isn't covered.
- Aiming at the centerline at 45° would need a different stop, such as the
  rear lane, or a steeper angle.
- Cross-track drift with the arm fixed: see the heading tolerance above.
