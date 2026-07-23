#!/usr/bin/env python3

import argparse
import math
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import yaml
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Static "world -> robot_map" transform published in scanning_entrypoint.sh
# (x=3.0 y=3.0 z=0.0, no rotation). Used to convert world-frame fixtures
# (walls, machine spawn pose) into the robot_map frame the plan file is in.
WORLD_TO_ROBOT_MAP = (3.0, 3.0, 0.0)

# Walls from renee_rbvogui_navigation/world/scanning.sdf, in world frame:
# (x, y, z, yaw, size_x, size_y, size_z).
WALLS_WORLD = [
    (4.0, 0.0, 1.0, 0.0, 0.1, 8.0, 2.0),
    (-4.0, 0.0, 1.0, 0.0, 0.1, 8.0, 2.0),
    (0.0, 4.0, 1.0, 1.5707963, 0.1, 8.0, 2.0),
    (0.0, -4.0, 1.0, -1.5707963, 0.1, 8.0, 2.0),
]

# Floor spans the room's interior (inner wall faces, i.e. wall centerline
# minus half the 0.1 m wall thickness), in robot_map: x in [-6.95, 0.95],
# y in [-6.95, 0.95].
FLOOR_BOUNDS_XY = ((-6.95, 0.95), (-6.95, 0.95))

# RB-Vogui+ chassis: rbvogui_body.urdf.xacro places chassis_link at
# xyz="-0.012 0 0.1775" from base_link, and base_footprint -> base_link adds
# wheel_radius=0.11 (rubber_wheel.urdf.xacro). The chassis_link inertial box
# is w=1.265 x h=0.812 x d=0.22, centered at chassis-local z=0.10, so the
# solid slab spans roughly z=[0.28, 0.50] above base_footprint/ground -- NOT
# from the ground up. The UR5e arm mounts on chassis_link at z=0.235, i.e.
# just above this slab's top, so drawing the box only across its real
# vertical extent keeps it visually distinct from the arm above it.
CHASSIS_SIZE = (1.265, 0.812, 0.22)
CHASSIS_Z_OFFSET = 0.28

# UR5e arm mount: chassis_link -> arm base is xyz="0 0 0.235" (rbvogui_plus
# arm_ur macro), on top of base_footprint -> base_link (wheel_radius=0.11) ->
# chassis_link (xyz="-0.012 0 0.1775"). So the arm's own base sits at
# (-0.012, 0, 0.5225) from base_pose, in the base's local (yawed) frame --
# not at ground level, which is where NavigateToPose's base_pose actually is.
ARM_BASE_LOCAL_XY_OFFSET = (-0.012, 0.0)
ARM_BASE_Z_OFFSET = 0.11 + 0.1775 + 0.235

# campetella_CRC bounding box, measured directly from its STL meshes (all
# parts are attached via zero-offset fixed joints, so the meshes are already
# positioned in a shared local frame) at the default robot_scale=0.001:
#   local min ~= (-2.8455, -1.2139, -0.2662) from campetella_base_link's origin
#   local max ~= ( 1.3751,  0.3470,  1.4475) from campetella_base_link's origin
# campetella_scan.yaml's machine.center is set to the *geometric center* of
# this box (spawn origin + the offset above), not the spawn origin itself --
# so the box is centered on machine_center with no further xy offset, and
# centered vertically on machine_center.z too.
MACHINE_SIZE = (4.2206, 1.5609, 1.7137)
MACHINE_CENTER_XY_OFFSET = (0.0, 0.0)                # box is centered on machine_center
MACHINE_Z_BOTTOM_OFFSET = -MACHINE_SIZE[2] / 2.0      # box is centered on machine_center.z too


def parse_position(pose_node):
    position = pose_node["position"]
    if len(position) != 3:
        raise ValueError("pose position must contain [x, y, z]")
    return [float(position[0]), float(position[1]), float(position[2])]


def parse_orientation(pose_node):
    orientation = pose_node.get("orientation", [0.0, 0.0, 0.0, 1.0])
    if len(orientation) != 4:
        raise ValueError("pose orientation must contain [x, y, z, w]")
    return [
        float(orientation[0]),
        float(orientation[1]),
        float(orientation[2]),
        float(orientation[3]),
    ]


def rotate_local_z_axis(quaternion):
    # tool0 convention (ur_macro.xacro flange-tool0 joint): X+ left, Y+ up,
    # Z+ front. The camera/TCP "forward" direction is the local Z axis.
    x, y, z, w = quaternion
    return [
        2.0 * (x * z + y * w),
        2.0 * (y * z - x * w),
        1.0 - 2.0 * (x * x + y * y),
    ]


def yaw_from_quaternion(quaternion):
    x, y, z, w = quaternion
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def arm_base_point(base_point, yaw):
    """Where the UR5e's own base actually is: on top of the chassis, not at
    base_pose's ground/base_footprint position."""
    local_x, local_y = ARM_BASE_LOCAL_XY_OFFSET
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    return [
        base_point[0] + local_x * cos_yaw - local_y * sin_yaw,
        base_point[1] + local_x * sin_yaw + local_y * cos_yaw,
        base_point[2] + ARM_BASE_Z_OFFSET,
    ]


def normalize(vector):
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 1e-12:
        return [0.0, 0.0, 0.0]
    return [value / norm for value in vector]


def load_waypoints(plan_file):
    with open(plan_file, "r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)

    if not data:
        raise ValueError(f"{plan_file} is empty")

    waypoints = data.get("scan_waypoints", [])
    if not isinstance(waypoints, list):
        raise ValueError("scan_waypoints must be a list")

    return data, waypoints


def infer_center(points):
    if not points:
        return None
    return [
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
        sum(point[2] for point in points) / len(points),
    ]


def set_axes_equal(ax):
    limits = [
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ]
    spans = [abs(limit[1] - limit[0]) for limit in limits]
    centers = [sum(limit) / 2.0 for limit in limits]
    radius = max(spans) / 2.0

    ax.set_xlim3d([centers[0] - radius, centers[0] + radius])
    ax.set_ylim3d([centers[1] - radius, centers[1] + radius])
    ax.set_zlim3d([centers[2] - radius, centers[2] + radius])


def box_edges(center_xy, z_bottom, size_xyz, yaw=0.0):
    """Return the 12 edges (as pairs of 3D points) of an axis-aligned-then-yawed box."""
    half_x, half_y, height = size_xyz[0] / 2.0, size_xyz[1] / 2.0, size_xyz[2]
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)

    corners_local = [
        (sx * half_x, sy * half_y, sz)
        for sx in (-1.0, 1.0)
        for sy in (-1.0, 1.0)
        for sz in (0.0, height)
    ]
    corners = []
    for lx, ly, lz in corners_local:
        wx = center_xy[0] + lx * cos_yaw - ly * sin_yaw
        wy = center_xy[1] + lx * sin_yaw + ly * cos_yaw
        wz = z_bottom + lz
        corners.append((wx, wy, wz))

    # corners indexed by (sx, sy, sz) in the order generated above:
    # 0:(-,-,bot) 1:(-,-,top) 2:(-,+,bot) 3:(-,+,top)
    # 4:(+,-,bot) 5:(+,-,top) 6:(+,+,bot) 7:(+,+,top)
    bottom = [0, 2, 6, 4]
    top = [1, 3, 7, 5]
    edges = []
    for ring in (bottom, top):
        for i in range(len(ring)):
            edges.append((corners[ring[i]], corners[ring[(i + 1) % len(ring)]]))
    for b, t in zip(bottom, top):
        edges.append((corners[b], corners[t]))
    return edges


def segments_to_nan_separated(segments):
    """Flatten (start, end) 3D segments into single x/y/z lists, NaN-separated,
    so they can be drawn with one Line3D artist instead of one per segment --
    each extra `ax.plot`/`ax.quiver` call is a separate artist that mplot3d has
    to re-project on every rotate/pan, which is what makes the plot feel slow
    once there are a few hundred of them (many small boxes + stick-figure
    lines add up fast)."""
    xs, ys, zs = [], [], []
    for (sx, sy, sz), (ex, ey, ez) in segments:
        xs += [sx, ex, math.nan]
        ys += [sy, ey, math.nan]
        zs += [sz, ez, math.nan]
    return xs, ys, zs


def plot_box(ax, center_xy, z_bottom, size_xyz, yaw=0.0, color="tab:gray", label=None):
    xs, ys, zs = segments_to_nan_separated(box_edges(center_xy, z_bottom, size_xyz, yaw))
    ax.plot(xs, ys, zs, color=color, linewidth=1.2, label=label)


def plot_plane(ax, corners, color="0.6", alpha=0.15):
    """Draw a single filled, semi-transparent quad given 4 (x, y, z) corners in order."""
    poly = Poly3DCollection([corners], facecolor=color, edgecolor=color, alpha=alpha)
    ax.add_collection3d(poly)


def floor_corners(bounds_xy, z=0.0):
    (x_min, x_max), (y_min, y_max) = bounds_xy
    return [
        (x_min, y_min, z),
        (x_max, y_min, z),
        (x_max, y_max, z),
        (x_min, y_max, z),
    ]


def wall_plane_corners(center_xy, z_bottom, length, height, yaw):
    """Corners of a wall's face, ignoring its thickness (a flat vertical plane)."""
    half_length = length / 2.0
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)

    def point(ly, z):
        return (
            center_xy[0] - ly * sin_yaw,
            center_xy[1] + ly * cos_yaw,
            z,
        )

    return [
        point(-half_length, z_bottom),
        point(half_length, z_bottom),
        point(half_length, z_bottom + height),
        point(-half_length, z_bottom + height),
    ]


def plot_waypoints(
    plan_file,
    save_file=None,
    show_labels=False,
    arrow_scale=0.25,
    machine_size=MACHINE_SIZE,
    machine_center_xy_offset=MACHINE_CENTER_XY_OFFSET,
    machine_z_bottom_offset=MACHINE_Z_BOTTOM_OFFSET,
    show_chassis=True,
    chassis_size=CHASSIS_SIZE,
    chassis_z_offset=CHASSIS_Z_OFFSET,
    show_walls=True,
):
    data, waypoints = load_waypoints(plan_file)

    base_points = []
    base_orientations = []
    ee_points = []
    ee_orientations = []
    labels = []

    for order, waypoint in enumerate(waypoints, start=1):
        labels.append(f"{order}: {waypoint.get('id', order - 1)}")
        base_points.append(parse_position(waypoint["base_pose"]))
        base_orientations.append(parse_orientation(waypoint["base_pose"]))
        ee_points.append(parse_position(waypoint["end_effector_pose"]))
        ee_orientations.append(parse_orientation(waypoint["end_effector_pose"]))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    if show_walls:
        plot_plane(ax, floor_corners(FLOOR_BOUNDS_XY), color="0.65", alpha=0.25)
        for wx, wy, wz, wyaw, _sx, sy, sz in WALLS_WORLD:
            center_x = wx - WORLD_TO_ROBOT_MAP[0]
            center_y = wy - WORLD_TO_ROBOT_MAP[1]
            z_bottom = (wz - WORLD_TO_ROBOT_MAP[2]) - sz / 2.0
            plot_plane(
                ax,
                wall_plane_corners((center_x, center_y), z_bottom, sy, sz, wyaw),
                color="0.4",
                alpha=0.25,
            )

    arm_base_points = [
        arm_base_point(base_point, yaw_from_quaternion(base_orientation))
        for base_point, base_orientation in zip(base_points, base_orientations)
    ]

    if arm_base_points:
        ax.plot(
            [point[0] for point in arm_base_points],
            [point[1] for point in arm_base_points],
            [point[2] for point in arm_base_points],
            marker="s",
            linestyle="none",
            color="tab:purple",
            label="UR5 base (on RB-Vogui)",
        )

    if ee_points:
        ax.plot(
            [point[0] for point in ee_points],
            [point[1] for point in ee_points],
            [point[2] for point in ee_points],
            marker="^",
            linestyle="none",
            label="End effector / camera",
        )

    # Stick-figure profile of the robot: a vertical stick from base_pose
    # (ground/base_footprint) up to where the arm actually mounts, then a
    # second stick from there to the end-effector/camera -- not a single
    # straight line from the ground straight to the camera, which isn't
    # where the arm's own base is. All waypoints are drawn as one artist
    # (NaN-separated) instead of 2-per-waypoint, for the same reason as
    # plot_box above.
    stick_segments = [
        segment
        for base_point, arm_base, ee_point in zip(base_points, arm_base_points, ee_points)
        for segment in ((base_point, arm_base), (arm_base, ee_point))
    ]
    if stick_segments:
        xs, ys, zs = segments_to_nan_separated(stick_segments)
        ax.plot(xs, ys, zs, color="0.7", linestyle="--", linewidth=0.8)

    # Camera/TCP forward direction (local Z of the end-effector pose), all
    # drawn in a single quiver call instead of one per waypoint.
    if ee_points:
        forwards = [normalize(rotate_local_z_axis(o)) for o in ee_orientations]
        ax.quiver(
            [p[0] for p in ee_points],
            [p[1] for p in ee_points],
            [p[2] for p in ee_points],
            [f[0] for f in forwards],
            [f[1] for f in forwards],
            [f[2] for f in forwards],
            length=arrow_scale,
            color="tab:red",
            normalize=True,
        )

    # Chassis slab only (see CHASSIS_Z_OFFSET comment above) -- it stops well
    # below where the arm actually mounts, so it can't be mistaken for a
    # static volume that includes the (moving) arm. A ring visits the same
    # base_pose once per height, so dedupe identical (x, y, yaw) base poses --
    # otherwise a 3-height ring draws (and has to re-project) the same box
    # 3 times over.
    if show_chassis:
        seen_base_poses = set()
        first = True
        for base_point, orientation in zip(base_points, base_orientations):
            yaw = yaw_from_quaternion(orientation)
            key = (round(base_point[0], 6), round(base_point[1], 6), round(yaw, 6))
            if key in seen_base_poses:
                continue
            seen_base_poses.add(key)
            plot_box(
                ax,
                (base_point[0], base_point[1]),
                base_point[2] + chassis_z_offset,
                chassis_size,
                yaw=yaw,
                color="tab:blue",
                label="RB-Vogui chassis (measured slab, arm mounts above it)" if first else None,
            )
            first = False

    center = data.get("machine_center")
    if center is None:
        center = infer_center(ee_points)
    if center is not None:
        ax.scatter(
            [center[0]],
            [center[1]],
            [center[2]],
            marker="x",
            color="black",
            s=80,
            label="machine_center (campetella spawn origin)",
        )
        plot_box(
            ax,
            (center[0] + machine_center_xy_offset[0], center[1] + machine_center_xy_offset[1]),
            center[2] + machine_z_bottom_offset,
            machine_size,
            yaw=0.0,
            color="tab:orange",
            label="Machine (measured from meshes)",
        )

    if show_labels:
        for label, ee_point in zip(labels, ee_points):
            ax.text(ee_point[0], ee_point[1], ee_point[2], label, fontsize=8)

    frame_id = data.get("frame_id", "unknown")
    ax.set_title(f"Generated scan waypoints ({frame_id})")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")

    handles, plot_labels = ax.get_legend_handles_labels()
    if show_walls:
        # Poly3DCollection planes don't reliably register with the automatic
        # legend, so add proxy patches for them.
        handles += [
            mpatches.Patch(color="0.65", alpha=0.25, label="Floor"),
            mpatches.Patch(color="0.4", alpha=0.25, label="Room walls"),
        ]
        plot_labels += ["Floor", "Room walls"]
    ax.legend(handles, plot_labels, fontsize=8)
    ax.grid(True)
    set_axes_equal(ax)

    if save_file:
        plt.savefig(save_file, dpi=160, bbox_inches="tight")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot scan waypoints generated by renee_planner, with the "
        "room walls and the campetella machine drawn to scale for a quick "
        "sanity check before running the real simulation."
    )
    parser.add_argument("plan_file", type=Path, help="Generated scan plan YAML file.")
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Save the plot to an image file instead of opening a window.",
    )
    parser.add_argument(
        "--labels",
        action="store_true",
        help="Show waypoint IDs next to end-effector points.",
    )
    parser.add_argument(
        "--arrow-scale",
        type=float,
        default=0.25,
        help="Length of camera orientation arrows.",
    )
    parser.add_argument(
        "--machine-size",
        type=float,
        nargs=3,
        default=MACHINE_SIZE,
        metavar=("X", "Y", "Z"),
        help="Machine bounding box size in meters (default: measured campetella_CRC, %(default)s).",
    )
    parser.add_argument(
        "--machine-center-offset",
        type=float,
        nargs=2,
        default=MACHINE_CENTER_XY_OFFSET,
        metavar=("DX", "DY"),
        help="Offset from machine_center to the box's horizontal center (default: %(default)s).",
    )
    parser.add_argument(
        "--machine-z-offset",
        type=float,
        default=MACHINE_Z_BOTTOM_OFFSET,
        help="Offset from machine_center.z to the bottom of the machine box (default: %(default)s).",
    )
    parser.add_argument(
        "--no-chassis",
        dest="show_chassis",
        action="store_false",
        help="Don't draw the RB-Vogui chassis slab at each base pose.",
    )
    parser.add_argument(
        "--chassis-size",
        type=float,
        nargs=3,
        default=CHASSIS_SIZE,
        metavar=("X", "Y", "Z"),
        help="Chassis slab bounding box size in meters (default: %(default)s).",
    )
    parser.add_argument(
        "--no-walls",
        dest="show_walls",
        action="store_false",
        help="Don't draw the room walls (from scanning.sdf).",
    )
    args = parser.parse_args()

    plot_waypoints(
        args.plan_file,
        save_file=args.save,
        show_labels=args.labels,
        arrow_scale=args.arrow_scale,
        machine_size=tuple(args.machine_size),
        machine_center_xy_offset=tuple(args.machine_center_offset),
        machine_z_bottom_offset=args.machine_z_offset,
        show_chassis=args.show_chassis,
        chassis_size=tuple(args.chassis_size),
        show_walls=args.show_walls,
    )


if __name__ == "__main__":
    main()
