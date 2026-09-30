"""Screw detection mission: the camera passes slowly over rows of screws/holes (no ROS).

Targets (`targets` in the experiment YAML, targets_frame):

  - {name, frame: camera, path: {start: [x, y, z], end: [x, y, z]}}

The line is the row of holes on the surface. The camera (a `line_path` task
on its optical frame, `tilt_deg: 90`, `roll: 0`) stays `standoff_m` straight
above the line, optical axis perpendicular to the surface and the image
aligned with the row, while the base drives slowly forward along its lane
(boat mode, no sideways velocity). Getting there is the same as a cleaning
pass (cleaning.py): Nav2 to the planned stop, a straight sideways shift
towards the machine if the arm needs it, MoveIt to the path start, then the
HQP pass.

Every `capture.spacing_m` along the line the pass stops: once the base is
still (base_lane static_speed for static_time_s), the `/capture_rgbd` action
(renee_action_servers) saves `capture.frame_count` RGB-D keyframes with the
robot and camera poses into <run folder>/captures; then the pass goes on.
This first version only records: detecting the holes/screws in the images
is a later step. captures.yaml lists every capture and its result.
"""
import os

import numpy as np
import yaml

from .cleaning import CleaningMission


class ScrewDetectionMission(CleaningMission):
    """Camera passes along lines (rows of holes), stopping to capture RGB-D keyframes."""

    name = "screw_detection"
    title = "Screw detection pass (camera above the line)"

    def load_targets(self, config: dict) -> list:
        """Read the path targets (cleaning.py) and the capture settings.

        Raises:
            ValueError: without the `capture:` section or with spacing_m <= 0.
        """
        capture = config.get("capture")
        if capture is None:
            raise ValueError("screw_detection needs the `capture:` section")
        self.capture_spacing_m = float(capture.get("spacing_m", 0.1))
        self.capture_frames = int(capture.get("frame_count", 3))
        if self.capture_spacing_m <= 0.0:
            raise ValueError("capture.spacing_m must be > 0")
        self.captures = []
        return super().load_targets(config)

    def _capture_points(self, length: float) -> list:
        """Output: path parameters (m) where the pass stops to capture, start to end."""
        points = list(np.arange(0.0, length, self.capture_spacing_m))
        # The last one just before the end (the pass step ends at the end).
        end = max(0.0, length - 0.005)
        if not points or end - points[-1] > 0.5 * self.capture_spacing_m:
            points.append(end)
        return points

    def _step_follow_path(self, io, step: dict, data: dict):
        """The cleaning pass, stopping at every capture point for /capture_rgbd."""
        task = self.path_task
        capture = data.get("capture")
        if capture is not None:
            return self._poll_capture(io, step, data, capture)
        result = super()._step_follow_path(io, step, data)
        if result is not None or not data.get("started"):
            return result
        if "points" not in data:
            data["points"] = self._capture_points(task.length)
        if data["points"] and task.s >= data["points"][0] - 1e-6:
            # Stop here: the path holds s, the base stops, the arm keeps its point.
            task.paused = True
            task.active = False
            self.base_task.stop_pass()
            self.hold_base()
            io.publish_twist(0.0, 0.0, 0.0)
            data["capture"] = {"s": data["points"].pop(0), "since": None, "call": None,
                               "stopped_at": io.now()}
        return None

    def _poll_capture(self, io, step: dict, data: dict, capture: dict):
        """Wait for the base to be still, capture, then resume the pass."""
        target = self.targets[step["target"]]
        now = io.now()
        io.publish_twist(0.0, 0.0, 0.0)
        if capture["call"] is None:
            vx, vy, wz = self.robot.dq[self.robot.base_v_slice]
            if max(np.hypot(vx, vy), abs(wz)) > self.base_task.static_speed:
                capture["since"] = None
                return None
            if capture["since"] is None:
                capture["since"] = now
            if now - capture["since"] < self.base_task.static_time_s:
                return None
            if not io.capture_ready():
                io.log.warning("Waiting for the /capture_rgbd action server", throttle_duration_sec=5.0)
                return None
            number = len([c for c in self.captures if c["target"] == target["name"]])
            capture["waypoint"] = f"{target['name']}_{number:03d}"
            capture["call"] = io.capture_rgbd(capture["waypoint"],
                                              os.path.join(self.run_dir or "/tmp", "captures"),
                                              self.capture_frames)
            return None
        result = capture["call"].poll()
        if result is None:
            return None
        frame = self.path_task
        self.captures.append({"target": target["name"], "waypoint": capture["waypoint"],
                              "s_m": round(float(capture["s"]), 3), "success": bool(result[0]),
                              "detail": result[1],
                              "camera_position_odom": [round(float(v), 4) for v in frame.last_frame_position]})
        message = (f"Target '{target['name']}': capture {capture['waypoint']} at s={capture['s']:.2f} m "
                   f"{'saved' if result[0] else 'failed'} ({result[1]})")
        # One call site per severity: rclpy raises if a call site changes severity.
        if result[0]:
            io.log.info(message)
        else:
            io.log.warning(message)
        # Resume the pass; the stop does not count against its timeout.
        data["start"] += now - capture["stopped_at"]
        data["capture"] = None
        frame.paused = False
        frame.active = True
        if self.posture_task is not None:
            self.posture_task.set_preferred(self.robot.q[self.robot.arm_q_slice].copy())
        self.base_task.start_pass(reverse=target["reverse"])
        self.reset_validity()
        return None

    def save_results(self) -> None:
        """The path results (cleaning.py) plus captures.yaml and a captures line in summary.txt."""
        super().save_results()
        if self.run_dir is None:
            return
        try:
            with open(os.path.join(self.run_dir, "captures.yaml"), "w") as handle:
                yaml.safe_dump({"captures": self.captures}, handle, sort_keys=False)
            saved = sum(1 for c in self.captures if c["success"])
            with open(os.path.join(self.run_dir, "summary.txt"), "a") as handle:
                handle.write(f"Captures: {saved}/{len(self.captures)} saved (captures.yaml)\n")
            self.log.info(f"Captures: {saved}/{len(self.captures)} saved")
        except OSError as exc:
            self.log.error(f"Could not write captures.yaml: {exc}")
