#!/usr/bin/env python3
"""Plot a mission run from its <mission>_run.npz (renee_trajectory_generation.recorders).

For every *_run.npz in the given run folders (or the given .npz files) writes
the figure next to it, as the same name with .png:

  camera poses (defect_detection): camera path, targets and results, base path
      and stops, tracking error, HQP residuals, arm joints, results table;
  path passes (cleaning, screw_detection): top view, tracking errors, path
      progress and base command, arm joints, HQP residuals.

Older runs (hqp_run.npz, <mission>_run.npz without the plan) are plotted too,
without what they did not save.

Usage:
    python3 legacy/scripts/plot_run.py data/defect_detection_sim/20260928_220437
    python3 legacy/scripts/plot_run.py data/*/2026092*          # several runs
"""
import argparse
import glob
import os
import sys

import numpy as np

# Plot colors per camera pose result status.
STATUS_COLORS = {
    "reached": "tab:green",
    "timeout": "tab:orange",
    "collision": "tab:red",
    "out_of_tolerance": "tab:cyan",
    "unreachable_ik": "tab:purple",
    "nav_failed": "tab:brown",
    "stow_failed": "tab:pink",
    "approach_failed": "tab:olive",
    "not_run": "tab:gray",
}


def _get(data, key, default=None):
    """Output: data[key] if the .npz has it, else default."""
    return data[key] if key in data.files else default


def _load_mplot3d() -> bool:
    """Register matplotlib's 3D projection, preferring its own mpl_toolkits.

    A venv matplotlib under ROS can pick up the system mpl_toolkits (a regular
    package that wins over the venv's namespace one), whose mplot3d does not
    match the running matplotlib.

    Output:
        bool: True if the "3d" projection is available.
    """
    import matplotlib
    toolkits = os.path.join(os.path.dirname(os.path.dirname(matplotlib.__file__)), "mpl_toolkits")
    try:
        import mpl_toolkits
        if os.path.isdir(toolkits):
            mpl_toolkits.__path__ = [toolkits]
        from matplotlib import projections
        from mpl_toolkits.mplot3d import Axes3D
        if "3d" not in projections.get_projection_names():
            projections.register_projection(Axes3D)
        return True
    except ImportError:
        return False


def plot_camera_poses(data, png_path: str) -> None:
    """Figure of a camera pose run (CameraPoseRecorder)."""
    import matplotlib.pyplot as plt
    has_3d = _load_mplot3d()
    dims = 3 if has_3d else 2

    t = data["time"]
    frame_position, pose_id = data["frame_position"], data["pose_id"]
    residual, failed, base, arm_q = data["residual"], data["failed"], data["base"], data["arm_q"]
    targets = _get(data, "result_target", np.zeros((0, 3)))
    look_at = _get(data, "result_look_at")
    cameras = _get(data, "result_camera_position", np.full(targets.shape, np.nan))
    status = [str(s) for s in _get(data, "result_status", [])]
    numbers = _get(data, "result_pose_id", np.arange(len(targets)))
    transitions = t[1:][np.diff(pose_id) != 0]

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(str(_get(data, "title", "Camera pose run")))

    # 1. Camera path, targets (x, with the aim) and where each pose ended (status
    # color); a top view if the 3D projection is not available.
    ax = fig.add_subplot(2, 3, 1, projection="3d" if has_3d else None)
    tracking = (pose_id >= 0) & np.all(np.isfinite(frame_position), axis=1)
    ax.plot(*np.where(tracking[:, None], frame_position, np.nan)[:, :dims].T, color="tab:blue", lw=1)
    for index, target in enumerate(targets):
        ax.scatter(*target[:dims], color="black", marker="x", s=60)
        ax.text(*target[:dims], f" {int(numbers[index]) + 1}")
        if look_at is not None:
            direction = 0.3 * (look_at[index] - target) / np.linalg.norm(look_at[index] - target)
            ax.quiver(*target[:dims], *direction[:dims], color="black", alpha=0.5,
                      **({} if has_3d else {"angles": "xy", "scale_units": "xy", "scale": 1}))
        if np.all(np.isfinite(cameras[index])):
            ax.scatter(*cameras[index][:dims], s=40, color=STATUS_COLORS.get(status[index], "tab:gray"))
    ax.set_title("Camera: target (x), result by status color" + ("" if has_3d else ", top view"))
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    points = np.vstack([frame_position[tracking], targets])
    points = points[np.all(np.isfinite(points), axis=1)]
    if not has_3d:
        ax.axis("equal")
        ax.grid(True)
    elif len(points):
        ax.set_zlabel("z [m]")
        center = (points.max(axis=0) + points.min(axis=0)) / 2.0
        radius = max(float(np.max(points.max(axis=0) - points.min(axis=0))) / 2.0, 0.2)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)

    # 2. Base path and stops.
    ax = fig.add_subplot(2, 3, 2)
    ax.plot(base[:, 0], base[:, 1], color="tab:blue")
    ax.plot(base[0, 0], base[0, 1], "go", label="start")
    ax.plot(base[-1, 0], base[-1, 1], "rs", label="end")
    for name, (x, y, yaw) in zip(_get(data, "station_name", []), _get(data, "station_base", np.zeros((0, 3)))):
        if not np.isfinite(x):
            continue
        ax.plot(x, y, "k^", ms=9)
        ax.arrow(x, y, 0.3 * np.cos(yaw), 0.3 * np.sin(yaw), head_width=0.06, color="k")
        ax.text(x, y, f"  {name}", fontsize=8)
    ax.set_title("Base trajectory and stops (odom)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.axis("equal")
    ax.grid(True)
    ax.legend()

    # 3. Tracking error.
    ax = fig.add_subplot(2, 3, 3)
    ax.plot(t, data["position_error"], label="position error [m]")
    ax.plot(t, data["aim_angle"], label="aim angle [rad]")
    for x in transitions:
        ax.axvline(x, color="gray", ls="--", lw=0.8)
    for key, color in (("reach_tolerance_m", "tab:blue"), ("reach_tolerance_rad", "tab:orange")):
        value = float(_get(data, key, np.nan))
        if np.isfinite(value):
            ax.axhline(value, color=color, ls=":", lw=0.8)
    ax.set_title("Camera tracking error (dashed: pose change)")
    ax.set_xlabel("t [s]")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    # 4. Optimization residual per level.
    ax = fig.add_subplot(2, 3, 4)
    for column, name in enumerate(data["level_names"]):
        ax.plot(t, np.where(residual[:, column] > 0, residual[:, column], np.nan), label=str(name))
    if failed.any():
        top = np.nanmax(residual) if np.isfinite(residual).any() else 1.0
        ax.plot(t[failed], np.full(failed.sum(), top), "rx", label="level failed")
    ax.set_title("HQP optimization residual per level")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("||A v - b||")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    # 5. Arm joints.
    ax = fig.add_subplot(2, 3, 5)
    for column in range(arm_q.shape[1]):
        ax.plot(t, arm_q[:, column], label=f"q{column + 1}")
    for x in transitions:
        ax.axvline(x, color="gray", ls="--", lw=0.8)
    ax.set_title("Arm joint trajectories")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("q [rad]")
    ax.grid(True)
    ax.legend(ncol=2)

    # 6. Results table.
    ax = fig.add_subplot(2, 3, 6)
    ax.axis("off")
    if len(targets):
        stations = _get(data, "result_station", [""] * len(targets))
        errors = _get(data, "result_position_error", np.full(len(targets), np.nan))
        aims = _get(data, "result_aim_angle", np.full(len(targets), np.nan))
        table = ax.table(
            colLabels=["pose", "stop", "target", "camera", "pos err [m]", "aim [rad]", "status"],
            cellText=[[int(numbers[i]) + 1, str(stations[i]), np.array2string(targets[i], precision=2),
                       np.array2string(cameras[i], precision=2), f"{errors[i]:.4f}", f"{aims[i]:.4f}",
                       status[i] if i < len(status) else ""] for i in range(len(targets))],
            colWidths=[0.06, 0.22, 0.19, 0.19, 0.11, 0.09, 0.14], loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1.0, 1.4)
        for index, name in enumerate(status, start=1):
            table[index, 6].set_facecolor(STATUS_COLORS.get(name, "white"))
            table[index, 6].set_alpha(0.4)
    ax.set_title("Camera poses")

    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    plt.close(fig)


def plot_path(data, png_path: str) -> None:
    """Figure of the line passes (PathRecorder)."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(str(_get(data, "title", "Path pass (line following)")))
    t = data["time"] - data["time"][0]

    ax = axes[0, 0]
    for index, (start, end) in enumerate(zip(_get(data, "path_start", []), _get(data, "path_end", []))):
        ax.plot([start[0], end[0]], [start[1], end[1]], "k--", lw=1, label="aim line" if index == 0 else None)
    ax.plot(data["target"][:, 0], data["target"][:, 1], color="tab:gray", lw=3, alpha=0.5, label="frame target")
    ax.plot(data["frame"][:, 0], data["frame"][:, 1], color="tab:blue", label="frame")
    ax.plot(data["base"][:, 0], data["base"][:, 1], color="tab:orange", label="base")
    ax.set_title("Top view (odom)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.axis("equal")
    ax.grid(True)
    ax.legend(loc="best", fontsize=8)

    ax = axes[0, 1]
    ax.plot(t, data["cross_track"] * 100, label="cross-track [cm]")
    ax.plot(t, data["position_error"] * 100, label="position error (incl. lag) [cm]")
    ax.plot(t, (data["standoff"] - np.nanmedian(data["standoff"])) * 100, label="standoff - median [cm]")
    ax.set_title("Tracking errors")
    ax.set_xlabel("t [s]")
    ax.grid(True)
    ax.legend(fontsize=8)

    ax = axes[0, 2]
    ax.plot(t, data["aim_angle"], color="tab:red")
    ax.set_title("Aim angle error [rad]")
    ax.set_xlabel("t [s]")
    ax.grid(True)

    ax = axes[1, 0]
    ax.plot(t, data["s"], label="s [m]")
    ax.plot(t, data["s_dot"] * 10, label="s_dot x10 [m/s]")
    for column, label in enumerate(("base vx x10 [m/s]", "base vy x10 [m/s]", "base wz x10 [rad/s]")):
        ax.plot(t, data["base_twist"][:, column] * 10, label=label)
    ax.set_title("Path progress and base command")
    ax.set_xlabel("t [s]")
    ax.grid(True)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    for joint in range(data["arm_q"].shape[1]):
        ax.plot(t, data["arm_q"][:, joint], label=f"q{joint + 1}")
    ax.set_title("Arm joints [rad]")
    ax.set_xlabel("t [s]")
    ax.grid(True)
    ax.legend(fontsize=8, ncol=2)

    ax = axes[1, 2]
    for index, level in enumerate(data["level_names"]):
        ax.semilogy(t, np.maximum(data["residual"][:, index], 1e-9), label=str(level))
    ax.set_title("HQP residual per level ||Av - b||")
    ax.set_xlabel("t [s]")
    ax.grid(True, which="both", alpha=0.4)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(png_path, dpi=110)
    plt.close(fig)


def plot_file(npz_path: str) -> str:
    """Plot one <mission>_run.npz next to it. Output: the PNG path."""
    data = np.load(npz_path)
    kind = str(_get(data, "kind", "camera_poses" if "pose_id" in data.files else "path"))
    png_path = os.path.splitext(npz_path)[0] + ".png"
    (plot_camera_poses if kind == "camera_poses" else plot_path)(data, png_path)
    return png_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", help="run folders (data/<experiment>/<stamp>) or *_run.npz files")
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    files = []
    for run in args.runs:
        files += [run] if run.endswith(".npz") else sorted(glob.glob(os.path.join(run, "*_run.npz")))
    if not files:
        print("no *_run.npz found in " + ", ".join(args.runs), file=sys.stderr)
        return 1
    for path in files:
        print(f"{path} -> {plot_file(path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
