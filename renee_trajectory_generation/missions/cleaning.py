"""Cleaning mission: non-contact nozzle passes along lines on the machine (no ROS).

Targets (`targets` in the experiment YAML, targets_frame):

  - {name, frame: nozzle, path: {start: [x, y, z], end: [x, y, z]},
     tilt_deg, standoff_m, reverse, arm_fixed}

The nozzle (a `line_path` task on that frame) aims at the line from
standoff_m, tilted tilt_deg above the horizontal towards the robot (the
target's own values, else the task's; 0 = horizontal, e.g. at a vertical
face), while the base drives along its lane (`base_lane` task: no sideways
velocity, no turning): forward, the line then taken in the loop's driving
direction, or with `reverse: true` backward, the line taken against it
(e.g. to come back along the lane without turning). The line is aimed at
from the loop's side. With `arm_fixed: true` the arm keeps the
configuration MoveIt reached at the path start for the whole pass (its
joints locked in the joint_limits task): only the base, driving its lane,
carries the frame along the line (the lane must then run parallel to it). The base stop is planned for the frame pose at the
path start; after the stop and MoveIt's motion there (mission.py), the
`follow_path` step runs the HQP pass until the path ends (done), or
collision (MoveIt predicts one) / timeout.

screw_detection.py runs the same passes with the camera.

Results: <mission>_run.npz (the pass samples and the paths; scripts/plot_run.py
draws them) and summary.txt.
"""
import math

import numpy as np

from ..mission import Mission
from ..recorders import PathRecorder, format_path_summary
from ..tasks import JointLimitsTask, LinePathTask

# A pass is aborted (timeout) after this many times its nominal duration.
PATH_TIMEOUT_FACTOR = 3.0


class CleaningMission(Mission):
    """Line path targets, run as HQP passes with the base driving its lane."""

    name = "cleaning"
    title = "Cleaning pass (nozzle along the line)"   # of the run data and its plot

    def load_targets(self, config: dict) -> list:
        """Read the path targets.

        Output:
            list[dict]: {"id", "name", "frame", "start", "end", "position",
                "look_at", "roll"} (position/look_at: the frame pose at the
                path start, for the base placement and MoveIt).

        Raises:
            ValueError: on a malformed target or without a line_path task on
                its frame.
        """
        self.path_task = next((t for t in self.all_tasks if isinstance(t, LinePathTask)), None)
        if self.path_task is None:
            raise ValueError(f"{self.name} needs a line_path task")
        task = self.path_task
        task.active = False  # enabled during a pass
        arm_joints = set(self.robot.arm_joint_names)
        self.arm_limits_task = next((t for t in self.all_tasks if isinstance(t, JointLimitsTask)
                                     and arm_joints <= set(t.joints)), None)
        targets = []
        for index, spec in enumerate(config.get("targets") or []):
            name = spec.get("name", f"target_{index + 1}")
            frame = self.frame_alias(config, spec.get("frame", ""))
            if frame != task.frame or "path" not in spec:
                raise ValueError(f"targets[{index}] '{name}': needs `path` and the line_path "
                                 f"task's frame {task.frame}")
            start = np.asarray(spec["path"]["start"], dtype=float)
            end = np.asarray(spec["path"]["end"], dtype=float)
            if start.shape != (3,) or end.shape != (3,):
                raise ValueError(f"targets[{index}] '{name}': path start/end must be [x, y, z]")
            reverse = bool(spec.get("reverse", False))
            arm_fixed = bool(spec.get("arm_fixed", False))
            if arm_fixed and self.arm_limits_task is None:
                raise ValueError(f"targets[{index}] '{name}': arm_fixed needs a joint_limits "
                                 "task on all the arm joints")
            tilt = math.radians(float(spec.get("tilt_deg", math.degrees(task.default_tilt))))
            standoff = float(spec.get("standoff_m", task.default_standoff_m))
            if not 0.0 <= tilt <= math.pi / 2 or standoff <= 0.0:
                raise ValueError(f"targets[{index}] '{name}': tilt_deg must be in [0, 90] "
                                 "and standoff_m > 0")
            # Along the driving direction (against it with reverse: the base
            # backs up) and aimed at from the loop's side of the line.
            middle = 0.5 * (start + end)
            nearest = self.loop.points[self.loop.nearest(*middle[:2])]
            heading = np.array([math.cos(nearest[2]), math.sin(nearest[2]), 0.0])
            if (float((end - start) @ heading) < 0.0) != reverse:
                self.log.info(f"Target '{name}': path reversed to the "
                              f"{'reverse ' if reverse else ''}driving direction")
                start, end = end, start
            side = nearest[:2] - middle[:2]
            side = side / max(np.linalg.norm(side), 1e-9)
            w = np.array([side[0] * math.cos(tilt), side[1] * math.cos(tilt), math.sin(tilt)])
            # Image y reference: the line direction for a steep view (as
            # LinePathTask.image_down()), so MoveIt, the planner and the HQP agree.
            direction = (end - start) / max(np.linalg.norm(end - start), 1e-9)
            targets.append({"id": index, "name": name, "frame": frame, "start": start, "end": end,
                            "position": start + standoff * w, "look_at": start,
                            "roll": task.roll, "tilt_deg": math.degrees(tilt), "standoff_m": standoff,
                            "reverse": reverse, "arm_fixed": arm_fixed,
                            "down": direction if tilt >= math.radians(60.0) else None})
        self.recorder = PathRecorder(self.level_tasks.keys())
        return targets

    def check_plan(self) -> None:
        """The base must stay clear of the machine along the whole pass, not only at its stop.

        Raises:
            ValueError: if the footprint at the pass end (the stop moved the
                path length along its heading) comes too close to a box.
        """
        for stop in self.stops:
            if stop.offset <= 0.0:
                continue  # on the loop, which is collision-free
            x, y, yaw = stop.pose
            for t in stop.targets:
                length = float(np.linalg.norm(self.targets[t]["end"] - self.targets[t]["start"]))
                if self.targets[t]["reverse"]:
                    length = -length  # the base backs up along its lane
                end = (x + length * math.cos(yaw), y + length * math.sin(yaw), yaw)
                if not self.planner.footprint_clear(*end):
                    raise ValueError(
                        f"Target '{self.targets[t]['name']}': at the pass end the base "
                        f"({end[0]:.2f}, {end[1]:.2f}), {stop.offset:.2f} m off its lane, comes closer "
                        f"than {self.planner.base_clearance_m} m to the machine; shorten the path")

    def target_steps(self, t: int, stop_number: int) -> list:
        return [{"type": "follow_path", "target": t, "stop": stop_number}]

    def _step_follow_path(self, io, step: dict, data: dict):
        """HQP pass along a path: base forward on its lane, the frame on the line."""
        target = self.targets[step["target"]]
        task = self.path_task
        if not data.get("started"):
            to_odom = io.to_odom()
            if to_odom is None:
                return None
            rotation, translation, _ = to_odom
            x, y, _ = self.base_pose()
            # Fixed in odom for the whole pass: localization corrections
            # would otherwise move the line under the frame.
            task.set_path(rotation @ target["start"] + translation,
                          rotation @ target["end"] + translation, (x, y),
                          standoff_m=target["standoff_m"], tilt_deg=target["tilt_deg"])
            # The line as run, in odom (the plot draws everything in odom).
            target["start_odom"], target["end_odom"] = task.aim_point(0.0), task.aim_point(task.length)
            task.paused = False
            task.active = True
            self.lock_arm(target["arm_fixed"])
            if self.posture_task is not None:
                # The arm keeps its start configuration; the base carries the motion.
                self.posture_task.set_preferred(self.robot.q[self.robot.arm_q_slice].copy())
            self.base_task.start_pass(reverse=target["reverse"])
            self.reset_validity()
            data.update(started=True, start=io.now(),
                        timeout=PATH_TIMEOUT_FACTOR * (task.length / task.speed + 2.0 * task.ramp_s))
            io.log.info(f"Target '{target['name']}': pass started ({task.length:.2f} m at "
                        f"{task.speed:.3f} m/s, tilt {target['tilt_deg']:.0f} deg, standoff "
                        f"{target['standoff_m']:.2f} m{', base in reverse' if target['reverse'] else ''}"
                        f"{', arm fixed' if target['arm_fixed'] else ''})")
            return None
        status, detail = self.hqp_step(io, task)
        if status == "ok":
            self.record_pass_sample(io, target["name"], task.s)
            self.recorder.record(io.now() - self._run_start, step["target"], task, self._last_reports,
                                 self.base_pose(), self._last_command.base_twist,
                                 self.robot.q[self.robot.arm_q_slice])
            io.log.info(f"Target '{target['name']}': s={task.s:.2f}/{task.length:.2f} m, cross-track "
                        f"{task.last_cross_track * 100:.1f} cm, aim {task.last_aim_angle:.3f} rad",
                        throttle_duration_sec=5.0)
        result = None
        if status == "collision":
            io.hold_arm(self.robot.q[self.robot.arm_q_slice])
            result = ("collision", detail)
        elif task.finished:
            result = ("done", "")
        elif io.now() - data["start"] > data["timeout"]:
            result = ("timeout", f"s={task.s:.2f}/{task.length:.2f} m")
        if result is None:
            return None
        task.paused = True
        task.active = False
        self.lock_arm(False)
        self.base_task.stop_pass()
        self.hold_base()
        self._result(target["id"], result[0], result[1])
        message = f"Target '{target['name']}': {result[0]}, covered {task.s:.2f}/{task.length:.2f} m"
        if result[0] == "done":
            io.log.info(message)
        else:
            io.log.warning(f"{message} ({result[1]})")
        return True, ""

    def lock_arm(self, locked: bool) -> None:
        """Lock (or free) the arm joints for a pass with the arm fixed."""
        self.path_task.along_only = locked
        if self.arm_limits_task is not None:
            self.arm_limits_task.locked = locked

    def record_result(self, t: int, status: str, detail: str = "") -> None:
        target = self.targets[t]
        self.recorder.add_path(target["name"], target.get("start_odom", target["start"]),
                               target.get("end_odom", target["end"]), status, detail)

    def save_results(self) -> None:
        """Write <mission>_run.npz and the results table (summary.txt)."""
        if self.run_dir is None:
            return
        if len(self.recorder):  # also for a pass interrupted before it ended
            try:
                path = self.recorder.save(self.run_dir, self.name, self.title)
                self.log.info(f"Run data written to {path} (plot: scripts/plot_run.py {self.run_dir})")
            except (OSError, ValueError) as exc:
                self.log.error(f"Could not write the run data: {exc}")
        if self.recorder.paths:
            self.write_summary([("Path targets", format_path_summary(self.recorder.summary()))])
