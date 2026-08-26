#!/usr/bin/env python3

import argparse
import math
from pathlib import Path
import subprocess
import warnings
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
import yaml

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


def structure_metadata(plan_data):
    """Read optional structure visualization settings embedded in a plan YAML."""
    structure = plan_data.get("structure")
    if structure is None:
        return None
    model = structure.get("model")
    pose = structure.get("origin_pose")
    if not isinstance(model, str) or not isinstance(pose, list) or len(pose) != 4:
        raise ValueError("plan structure must contain model and origin_pose [x, y, z, yaw]")
    args = structure.get("xacro_args", [])
    if not isinstance(args, list) or not all(isinstance(argument, str) for argument in args):
        raise ValueError("plan structure.xacro_args must be a list of strings")
    return Path(model), tuple(float(value) for value in pose), tuple(args)


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


def rpy_matrix(roll, pitch, yaw):
    """Rotation matrix for URDF's fixed-axis roll, pitch, yaw convention."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def transform_matrix(xyz=(0.0, 0.0, 0.0), rpy=(0.0, 0.0, 0.0)):
    transform = np.eye(4)
    transform[:3, :3] = rpy_matrix(*rpy)
    transform[:3, 3] = xyz
    return transform


def parse_vector(value, length, label):
    if value is None:
        return [0.0] * length
    values = [float(item) for item in value.split()]
    if len(values) != length:
        raise ValueError(f"{label} must contain {length} values")
    return values


def origin_transform(node):
    origin = node.find("origin")
    if origin is None:
        return np.eye(4)
    return transform_matrix(
        parse_vector(origin.get("xyz"), 3, "origin xyz"),
        parse_vector(origin.get("rpy"), 3, "origin rpy"),
    )


def expand_robot_description(model_path, xacro_args):
    if model_path.suffix != ".xacro":
        return model_path.read_text(encoding="utf-8")
    command = ["xacro", str(model_path), *xacro_args]
    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True)
    except FileNotFoundError as error:
        raise RuntimeError("xacro is required to plot a .xacro model; source the ROS workspace first") from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"xacro failed for {model_path}: {error.stderr.strip()}") from error
    return result.stdout


def primitive_from_collision(collision, link_transform):
    geometry = collision.find("geometry")
    if geometry is None:
        return None, False
    transform = link_transform @ origin_transform(collision)
    box = geometry.find("box")
    if box is not None:
        return {"kind": "box", "size": parse_vector(box.get("size"), 3, "box size"), "transform": transform}, False
    cylinder = geometry.find("cylinder")
    if cylinder is not None:
        return {"kind": "cylinder", "radius": float(cylinder.get("radius")), "length": float(cylinder.get("length")), "transform": transform}, False
    sphere = geometry.find("sphere")
    if sphere is not None:
        return {"kind": "sphere", "radius": float(sphere.get("radius")), "transform": transform}, False
    return None, geometry.find("mesh") is not None


def load_structure_primitives(model_path, structure_pose, xacro_args=()):
    """Load primitive collision geometry from any URDF or expanded Xacro."""
    xml_text = expand_robot_description(model_path, xacro_args)
    try:
        robot = ET.fromstring(xml_text)
    except ET.ParseError as error:
        raise ValueError(f"Invalid URDF/XML in {model_path}: {error}") from error

    links = {link.get("name"): link for link in robot.findall("link") if link.get("name")}
    children = set()
    joints_by_parent = {}
    for joint in robot.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None or not parent.get("link") or not child.get("link"):
            continue
        children.add(child.get("link"))
        joints_by_parent.setdefault(parent.get("link"), []).append((child.get("link"), origin_transform(joint)))

    roots = [name for name in links if name not in children]
    if not roots:
        raise ValueError("URDF has no root link")
    x, y, z, yaw = structure_pose
    root_transform = transform_matrix((x, y, z), (0.0, 0.0, yaw))
    primitives, mesh_count, visited = [], 0, set()

    def visit(link_name, link_transform):
        nonlocal mesh_count
        if link_name in visited:
            return
        visited.add(link_name)
        link = links.get(link_name)
        if link is not None:
            for collision in link.findall("collision"):
                primitive, has_mesh = primitive_from_collision(collision, link_transform)
                if primitive is not None:
                    primitives.append(primitive)
                mesh_count += int(has_mesh)
        for child_name, joint_transform in joints_by_parent.get(link_name, []):
            visit(child_name, link_transform @ joint_transform)

    for root in roots:
        visit(root, root_transform)
    if not primitives:
        suffix = "; mesh collisions are not rendered" if mesh_count else ""
        raise ValueError(f"No supported collision primitives (box, cylinder, sphere) found in {model_path}{suffix}")
    if mesh_count:
        warnings.warn(f"Skipped {mesh_count} mesh collision(s) in {model_path}; use simple primitive collisions for the basic structure plot.", RuntimeWarning)
    return primitives


def transform_points(transform, points):
    local = np.asarray(points, dtype=float)
    homogeneous = np.column_stack((local, np.ones(len(local))))
    return (transform @ homogeneous.T).T[:, :3]


def plot_structure(ax, primitives):
    """Draw basic URDF collision primitives as lightweight wireframes."""
    first = True
    for primitive in primitives:
        color = "tab:orange"
        label = "Structure collision primitives" if first else None
        first = False
        transform = primitive["transform"]
        if primitive["kind"] == "box":
            sx, sy, sz = (value / 2.0 for value in primitive["size"])
            corners = [(a * sx, b * sy, c * sz) for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)]
            edges = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
            points = transform_points(transform, corners)
            segments = [(points[a], points[b]) for a, b in edges]
            xs, ys, zs = segments_to_nan_separated(segments)
            ax.plot(xs, ys, zs, color=color, linewidth=0.8, label=label)
        elif primitive["kind"] == "cylinder":
            angles = np.linspace(0.0, 2.0 * math.pi, 20)
            radius, half_length = primitive["radius"], primitive["length"] / 2.0
            bottom = [(radius * math.cos(a), radius * math.sin(a), -half_length) for a in angles]
            top = [(radius * math.cos(a), radius * math.sin(a), half_length) for a in angles]
            for ring, ring_label in ((bottom, label), (top, None)):
                points = transform_points(transform, ring)
                ax.plot(points[:, 0], points[:, 1], points[:, 2], color=color, linewidth=0.8, label=ring_label)
            bottom_points, top_points = transform_points(transform, bottom), transform_points(transform, top)
            for index in range(0, len(angles), 5):
                ax.plot([bottom_points[index, 0], top_points[index, 0]], [bottom_points[index, 1], top_points[index, 1]], [bottom_points[index, 2], top_points[index, 2]], color=color, linewidth=0.8)
        else:
            radius = primitive["radius"]
            angles = np.linspace(0.0, 2.0 * math.pi, 24)
            rings = [[(radius * math.cos(a), radius * math.sin(a), 0.0) for a in angles], [(radius * math.cos(a), 0.0, radius * math.sin(a)) for a in angles], [(0.0, radius * math.cos(a), radius * math.sin(a)) for a in angles]]
            for ring_index, ring in enumerate(rings):
                points = transform_points(transform, ring)
                ax.plot(points[:, 0], points[:, 1], points[:, 2], color=color, linewidth=0.8, label=label if ring_index == 0 else None)


def plot_waypoints(
    plan_file,
    save_file=None,
    show_labels=False,
    arrow_scale=0.25,
    structure_model=None,
    structure_pose=None,
    xacro_args=(),
    show_chassis=True,
    chassis_size=CHASSIS_SIZE,
    chassis_z_offset=CHASSIS_Z_OFFSET,
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

    configured_structure = structure_metadata(data)
    model = Path(structure_model) if structure_model is not None else None
    pose = tuple(structure_pose) if structure_pose is not None else None
    arguments = tuple(xacro_args)
    if configured_structure is not None:
        config_model, config_pose, config_arguments = configured_structure
        model = model or config_model
        pose = pose or config_pose
        arguments = arguments or config_arguments
    if model is not None:
        if pose is None:
            raise ValueError("--structure-pose is required when --structure-model is used and the plan has no structure origin_pose")
        plot_structure(ax, load_structure_primitives(model, pose, arguments))
    else:
        warnings.warn("No structure model in the plan or CLI; plotting waypoints without structure geometry.", RuntimeWarning)

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
            label="machine_center (from plan)",
        )

    if show_labels:
        for label, ee_point in zip(labels, ee_points):
            ax.text(ee_point[0], ee_point[1], ee_point[2], label, fontsize=8)

    frame_id = data.get("frame_id", "unknown")
    ax.set_title(f"Generated scan waypoints ({frame_id})")

    handles, plot_labels = ax.get_legend_handles_labels()
    ax.legend(handles, plot_labels, fontsize=8)
    ax.set_axis_off()
    set_axes_equal(ax)

    if save_file:
        plt.savefig(save_file, dpi=160, bbox_inches="tight")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot scan waypoints generated by renee_planner, with "
        "optional generic URDF/Xacro collision primitives for a quick "
        "sanity check before running the simulation."
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
        "--structure-model", type=Path, default=None,
        help="URDF or Xacro model whose collision primitives will be drawn.",
    )
    parser.add_argument(
        "--structure-pose", type=float, nargs=4, default=None,
        metavar=("X", "Y", "Z", "YAW"),
        help="Pose of the model origin in the plan frame; required with --structure-model.",
    )
    parser.add_argument(
        "--xacro-arg", action="append", default=[], metavar="NAME:=VALUE",
        help="Xacro argument, repeatable; ignored for URDF input.",
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
    args = parser.parse_args()

    plot_waypoints(
        args.plan_file,
        save_file=args.save,
        show_labels=args.labels,
        arrow_scale=args.arrow_scale,
        structure_model=args.structure_model,
        structure_pose=tuple(args.structure_pose) if args.structure_pose else None,
        xacro_args=tuple(args.xacro_arg),
        show_chassis=args.show_chassis,
        chassis_size=tuple(args.chassis_size),
    )


if __name__ == "__main__":
    main()
