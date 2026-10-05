"""Run recorders: the missions' per-tick data and results, saved as <mission>_run.npz (no plotting).

``CameraPoseRecorder`` (defect_detection) collects one sample per control tick
(camera position and errors, per-level solver reports, base and arm state),
the plan (camera poses, base stops) and how each pose ended.
``PathRecorder`` (cleaning, screw_detection) collects one sample per tick of a
pass (line progress, tracked frame and its target, errors, base and arm) and
how each path ended.

save() writes everything, plan and results included, to <mission>_run.npz;
scripts/plot_run.py draws the figures from it. The format_*() tables go to
summary.txt.
"""

import os

import numpy as np


def _strings(values) -> np.ndarray:
    """Output: a unicode array (loadable without allow_pickle), also when empty."""
    return np.asarray([str(v) for v in values], dtype=str)


class CameraPoseRecorder:
    """Accumulates per-tick samples and per-pose results of a camera pose run.

    The mission declares the plan (camera poses, grouped by base stop) with
    set_plan(), records one sample per control tick with record(), and
    reports how each pose ended with add_result().
    """

    kind = "camera_poses"

    def __init__(self, level_names, reach_tolerance_m: float = float("nan"),
                 reach_tolerance_rad: float = float("nan")) -> None:
        """Prepare empty buffers for one run.

        Input:
            level_names: HQP level names in priority order, used to index
                per-level residuals and solve times.
            reach_tolerance_m: position tolerance (drawn on the error plot).
            reach_tolerance_rad: aim tolerance (drawn on the error plot).
        """
        self.level_names = list(level_names)
        self.reach_tolerance_m = reach_tolerance_m
        self.reach_tolerance_rad = reach_tolerance_rad
        self.poses = []      # [{"pose_id", "station", "position", "look_at"}] in odom
        self.stations = []   # [{"name", "base": (x, y, yaw) in odom or None}]
        self.results = {}    # pose_id -> result dict (see add_result)
        self.time = []
        self.frame_position = []
        self.position_error = []
        self.aim_angle = []
        self.pose_id = []
        self.residual = []
        self.solve_time = []
        self.failed = []
        self.base = []
        self.arm_q = []

    def __len__(self) -> int:
        """Output: int, the number of recorded ticks."""
        return len(self.time)

    def set_plan(self, poses: list, stations: list) -> None:
        """Declare every camera pose and base stop of the run (in odom).

        Input:
            poses: list of {"pose_id", "station", "position", "look_at"}.
            stations: list of {"name", "base"}, base = (x, y, yaw) or None.
        """
        self.poses = list(poses)
        self.stations = list(stations)

    def record(self, t: float, frame_position, reports, base_xy_yaw, arm_q,
               position_error: float = float("nan"), aim_angle: float = float("nan"),
               pose_id: int = -1) -> None:
        """Append one control tick.

        Input:
            t: time since the run started (s).
            frame_position: tracked camera position in odom, (3,).
            reports: list of solver.LevelReport from this tick's step().
            base_xy_yaw: base (x, y, yaw) in odom.
            arm_q: arm joint positions (rad), (6,).
            position_error: camera position error (m) for the active pose.
            aim_angle: camera aim error (rad) for the active pose.
            pose_id: active camera pose, -1 if none.
        """
        by_name = {r.name: r for r in reports}
        self.time.append(float(t))
        self.frame_position.append(np.asarray(frame_position, dtype=float))
        self.position_error.append(float(position_error))
        self.aim_angle.append(float(aim_angle))
        self.pose_id.append(int(pose_id))
        self.residual.append(
            [by_name[n].residual_norm if n in by_name else np.nan for n in self.level_names])
        self.solve_time.append(
            [by_name[n].solve_time_s if n in by_name else np.nan for n in self.level_names])
        self.failed.append(any(r.status != "solved" for r in reports))
        self.base.append(np.asarray(base_xy_yaw, dtype=float))
        self.arm_q.append(np.asarray(arm_q, dtype=float))

    def add_result(self, pose_id: int, status: str, camera_position=None,
                   position_error: float = float("nan"), aim_angle: float = float("nan"),
                   detail: str = "") -> None:
        """Store how one camera pose ended.

        Input:
            pose_id: pose from set_plan().
            status: reached | timeout | collision | out_of_tolerance | unreachable_ik |
                nav_failed | ...
            camera_position: camera position (odom) when the pose ended, if any.
            position_error: final position error (m).
            aim_angle: final aim error (rad).
            detail: free text for the log (e.g. collision contacts).
        """
        self.results[pose_id] = {
            "status": status,
            "camera_position": (np.asarray(camera_position, dtype=float)
                                if camera_position is not None else np.full(3, np.nan)),
            "position_error": float(position_error),
            "aim_angle": float(aim_angle),
            "detail": detail,
        }

    def summary(self) -> list:
        """One row per planned camera pose, in plan order.

        Output:
            list[dict]: pose_id, station, target, look_at, camera_position,
                position_error, aim_angle, status ("not_run" if the run ended
                before the pose), detail.
        """
        rows = []
        for pose in self.poses:
            result = self.results.get(pose["pose_id"], {
                "status": "not_run", "camera_position": np.full(3, np.nan),
                "position_error": float("nan"), "aim_angle": float("nan"), "detail": ""})
            rows.append({"pose_id": pose["pose_id"], "station": pose["station"],
                         "target": np.asarray(pose["position"]),
                         "look_at": np.asarray(pose["look_at"]), **result})
        return rows

    def save(self, out_dir: str, name: str, title: str = "") -> str:
        """Write the samples, the plan and the results to <out_dir>/<name>_run.npz.

        Output:
            str: path of the written file.

        Raises:
            ValueError: if nothing was recorded.
        """
        if not len(self):
            raise ValueError("No samples recorded")
        os.makedirs(out_dir, exist_ok=True)
        summary = self.summary()
        path = os.path.join(out_dir, f"{name}_run.npz")
        np.savez(
            path, kind=self.kind, title=title or name,
            time=np.asarray(self.time), frame_position=np.vstack(self.frame_position),
            position_error=np.asarray(self.position_error), aim_angle=np.asarray(self.aim_angle),
            pose_id=np.asarray(self.pose_id), residual=np.asarray(self.residual, dtype=float),
            solve_time=np.asarray(self.solve_time, dtype=float), failed=np.asarray(self.failed),
            base=np.vstack(self.base), arm_q=np.vstack(self.arm_q),
            level_names=_strings(self.level_names),
            reach_tolerance_m=self.reach_tolerance_m, reach_tolerance_rad=self.reach_tolerance_rad,
            result_pose_id=np.asarray([row["pose_id"] for row in summary], dtype=int),
            result_station=_strings(row["station"] for row in summary),
            result_status=_strings(row["status"] for row in summary),
            result_detail=_strings(row["detail"] for row in summary),
            result_target=np.asarray([row["target"] for row in summary], dtype=float).reshape(-1, 3),
            result_look_at=np.asarray([row["look_at"] for row in summary], dtype=float).reshape(-1, 3),
            result_camera_position=np.asarray([row["camera_position"] for row in summary],
                                              dtype=float).reshape(-1, 3),
            result_position_error=np.asarray([row["position_error"] for row in summary], dtype=float),
            result_aim_angle=np.asarray([row["aim_angle"] for row in summary], dtype=float),
            station_name=_strings(s["name"] for s in self.stations),
            station_base=np.asarray([s["base"] if s["base"] is not None else (np.nan,) * 3
                                     for s in self.stations], dtype=float).reshape(-1, 3))
        return path


def format_camera_summary(summary: list) -> str:
    """Render CameraPoseRecorder.summary() as a plain-text table.

    Output:
        str: one line per camera pose, empty if summary is empty.
    """
    if not summary:
        return ""
    lines = ["pose  station                target               camera               "
             "pos_err[m]  aim[rad]  status"]
    for row in summary:
        target = ", ".join(f"{v:.2f}" for v in row["target"])
        camera = ", ".join(f"{v:.2f}" for v in row["camera_position"])
        lines.append(
            f"{row['pose_id'] + 1:>4}  {row['station'][:21]:<21}  [{target}]  [{camera}]  "
            f"{row['position_error']:>10.4f}  {row['aim_angle']:>8.4f}  {row['status']}"
            + (f" ({row['detail']})" if row["detail"] else ""))
    return "\n".join(lines)


class PathRecorder:
    """Per-tick samples of the line passes (LinePathTask) and how each path ended."""

    kind = "path"
    FIELDS = ("time", "path", "s", "s_dot", "frame", "target", "position_error",
              "cross_track", "standoff", "aim_angle", "base", "base_twist", "arm_q")

    def __init__(self, level_names) -> None:
        """Input: level_names, HQP level names in priority order."""
        self.level_names = list(level_names)
        self.samples = {field: [] for field in self.FIELDS}
        self.residual = []
        self.paths = []  # [{"name", "start", "end", "status", "detail"}] in odom

    def __len__(self) -> int:
        """Output: int, number of recorded samples."""
        return len(self.samples["time"])

    def record(self, t: float, path: int, task, reports, base_xy_yaw, base_twist, arm_q) -> None:
        """Store one tick of a pass.

        Input:
            t: time (s) since the run start.
            path: index of the active path target.
            task: LinePathTask after this tick's build().
            reports: solver.LevelReport list of this tick.
            base_xy_yaw: (x, y, yaw) of the base in odom.
            base_twist: commanded (vx, vy, wz) in the base frame.
            arm_q: (6,) arm joint positions.
        """
        values = (t, path, task.s, task.s_dot, task.last_frame_position, task.last_target,
                  task.last_position_error, task.last_cross_track, task.last_standoff,
                  task.last_aim_angle, base_xy_yaw, base_twist, arm_q)
        for field, value in zip(self.FIELDS, values):
            self.samples[field].append(np.array(value, dtype=float))
        residual = np.full(len(self.level_names), np.nan)
        for report in reports:
            if report.name in self.level_names:
                residual[self.level_names.index(report.name)] = report.residual_norm
        self.residual.append(residual)

    def add_path(self, name: str, start, end, status: str, detail: str = "") -> None:
        """Store how a path ended (status: done | collision | timeout | ...)."""
        self.paths.append({"name": name, "start": np.asarray(start, dtype=float),
                           "end": np.asarray(end, dtype=float), "status": status, "detail": detail})

    def summary(self) -> list:
        """Output: list[dict] per path: name, status, samples and error maxima."""
        rows = []
        path_ids = np.asarray(self.samples["path"]) if len(self) else np.zeros(0)
        for index, path in enumerate(self.paths):
            mask = path_ids == index
            row = {"name": path["name"], "status": path["status"], "detail": path["detail"],
                   "samples": int(mask.sum())}
            for field in ("position_error", "cross_track", "aim_angle"):
                values = np.asarray(self.samples[field])[mask] if mask.any() else np.zeros(0)
                row[f"max_{field}"] = float(np.nanmax(values)) if values.size else float("nan")
            standoff = np.asarray(self.samples["standoff"])[mask] if mask.any() else np.zeros(0)
            row["standoff_range"] = ((float(np.nanmin(standoff)), float(np.nanmax(standoff)))
                                     if standoff.size else (float("nan"), float("nan")))
            twist = np.vstack(self.samples["base_twist"])[mask] if mask.any() else np.zeros((0, 3))
            row["max_abs_vy"] = float(np.max(np.abs(twist[:, 1]))) if len(twist) else float("nan")
            s = np.asarray(self.samples["s"])[mask] if mask.any() else np.zeros(0)
            row["covered_m"] = float(s.max()) if s.size else 0.0
            row["length_m"] = float(np.linalg.norm(path["end"] - path["start"]))
            rows.append(row)
        return rows

    def save(self, out_dir: str, name: str, title: str = "") -> str:
        """Write the samples and the paths to <out_dir>/<name>_run.npz.

        Output:
            str: path of the written file.

        Raises:
            ValueError: if nothing was recorded.
        """
        if not len(self):
            raise ValueError("No path samples recorded")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{name}_run.npz")
        np.savez(
            path, kind=self.kind, title=title or name,
            residual=np.asarray(self.residual, dtype=float), level_names=_strings(self.level_names),
            path_name=_strings(p["name"] for p in self.paths),
            path_status=_strings(p["status"] for p in self.paths),
            path_detail=_strings(p["detail"] for p in self.paths),
            path_start=np.asarray([p["start"] for p in self.paths], dtype=float).reshape(-1, 3),
            path_end=np.asarray([p["end"] for p in self.paths], dtype=float).reshape(-1, 3),
            **{field: np.asarray(values) for field, values in self.samples.items()})
        return path


def format_path_summary(rows: list) -> str:
    """Output: str, one line per path target (for the log and summary.txt)."""
    return "\n".join(
        f"  {row['name']}: {row['status']}{' (' + row['detail'] + ')' if row['detail'] else ''}, "
        f"covered {row['covered_m']:.2f}/{row['length_m']:.2f} m, max cross-track "
        f"{row['max_cross_track'] * 100:.1f} cm, max pos err {row['max_position_error'] * 100:.1f} cm, "
        f"max aim {row['max_aim_angle']:.3f} rad, standoff {row['standoff_range'][0]:.3f}-"
        f"{row['standoff_range'][1]:.3f} m, max |vy| {row['max_abs_vy']:.4f} m/s"
        for row in rows)
