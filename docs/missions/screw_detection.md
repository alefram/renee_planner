# Screw detection mission (`screw_detection_sim`)

The wrist ZED 2i camera passes slowly **straight above** the row of holes on
the rail's top face. These holes are where the screws will go. The pass
stops every `capture.spacing_m` to save RGB-D keyframes. This first version
only records the data; detecting the holes and screws in the images is a
later step.

## The motion

- **Target**: `rail_holes`, the top-face centerline y -3.085, z 0.87, from
  x -2.93 to -3.80 (0.87 m, between the front supports).
- **Camera**: the optical frame is 0.30 m straight above the line
  (`tilt_deg: 90`), which is the ZED 2i depth minimum. The optical axis is
  perpendicular to the surface, and the image is aligned with the row
  (`roll: 3.14159`).
- **Base**: Nav2 drives to the stop, the base shifts straight toward the
  machine if the arm needs it, and MoveIt places the camera. The base then
  drives slowly forward along the lane (vy = 0) while the HQP keeps the
  camera on the line.
- **Captures**: every 0.1 m, the pass stops. Once the base has been still
  for `static_time_s`, the `/capture_rgbd` action (`renee_action_servers`)
  saves 3 keyframes with the robot and camera poses, then the pass resumes.
  The capture server is started by the launch file
  (`capture.start_server: true`).

The mission class is `ScrewDetectionMission(CleaningMission)`. It reuses the
cleaning pass and adds the capture stops (`missions/screw_detection.py`).

## Latest result (2026-09-26, `data/screw_detection_sim/20260926_212847`)

- Covered 0.87 / 0.87 m, max cross-track 0.3 cm, standoff 0.298–0.300 m,
  max aim error 0.001 rad, max vy 0.
- 10 / 10 captures saved (`captures.yaml`, `captures/`).
- Needed fix: an IMU system plugin in `renee_rbvogui_navigation/world/scanning.sdf`,
  otherwise the ZED's IMU topic was empty.
- The capture sync tolerance is 300 ms, because simulated color, right and
  depth frames arrive about 130 ms apart.
- Video: `corner_video_from34s.mp4` (the original, trimmed to start at 34 s).
