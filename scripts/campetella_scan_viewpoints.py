#!/usr/bin/env python3
"""Camera viewpoints around the Campetella, editable in web_tf_editor.

publish: generates viewpoints bordering the machine (camera optical frame:
+Z aimed at a point on the machine's surface, image upright, `--distance` m
away) and publishes them to web_tf_editor's frame_bridge_node, so they show
up as editable frames in the browser and on /tf.

export: reads the (possibly corrected) frames back from frame_bridge_node and
prints them as `targets:` entries for config/experiments/defect_detection_sim.yaml
(position = frame origin, look_at = the point `--distance` m along its +Z).
Frames whose look_at distance is outside [--min, --max] are flagged.

load: publishes an experiment YAML's camera targets (--experiment, e.g.
config/experiments/defect_detection_sim.yaml) as the same frames, to monitor
or correct the poses the experiment will run.

Geometry (robot_map, the Campetella's CAD bounds): rail along X
(x -4.85..-1.68, y -3.27..-2.90, z 0.60-0.87), column (x ~-2.1, y ~-2.73,
z 0.53-2.25), extraction axis along Y (x ~-2.0, y -4.21..-2.81, z 0.8-1.15).

  ros2 run ... / python3 campetella_scan_viewpoints.py publish [--distance 0.75]
  python3 campetella_scan_viewpoints.py export [--distance 0.75]
"""
import argparse
import json
import sys
import time

from geometry_msgs.msg import TransformStamped
import numpy as np
import rclpy
from rclpy.qos import QoSDurabilityPolicy, QoSProfile
from std_msgs.msg import String

FRAME = "robot_map"
PREFIX = "scan_"

# (name, look_at on the machine surface, direction from look_at towards the camera)
VIEWS = [
    # Rail, front side (+Y), -X to +X.
    ("rail_front_1", (-4.40, -2.90, 0.75), (0, 1, 0.45)),
    ("rail_front_2", (-3.60, -2.90, 0.75), (0, 1, 0.45)),
    ("rail_front_3", (-2.80, -2.90, 0.75), (0, 1, 0.45)),
    # Column, front side.
    ("column_front", (-2.10, -2.73, 1.20), (0, 1, 0.15)),
    # Rail, +X end.
    ("rail_end_pos_x", (-1.68, -3.08, 0.75), (1, 0, 0.45)),
    # Extraction axis, +X side.
    ("extraction_pos_x_1", (-1.95, -3.40, 0.95), (1, 0, 0.35)),
    ("extraction_pos_x_2", (-1.95, -3.90, 0.95), (1, 0, 0.35)),
    # Extraction axis, rear end (-Y).
    ("extraction_rear", (-2.00, -4.21, 0.95), (0, -1, 0.35)),
    # Extraction axis, -X side.
    ("extraction_neg_x", (-2.10, -3.90, 0.95), (-1, 0, 0.35)),
    # Rail, rear side (-Y), +X to -X.
    ("rail_rear_1", (-2.80, -3.27, 0.75), (0, -1, 0.45)),
    ("rail_rear_2", (-3.60, -3.27, 0.75), (0, -1, 0.45)),
    ("rail_rear_3", (-4.40, -3.27, 0.75), (0, -1, 0.45)),
    # Rail, -X end.
    ("rail_end_neg_x", (-4.85, -3.08, 0.75), (-1, 0, 0.45)),
]


def optical_quaternion(z):
    """Optical frame looking along z: +Y down in the image (upright), +X right."""
    z = z / np.linalg.norm(z)
    down = np.array([0.0, 0.0, -1.0])
    y = down - z * np.dot(down, z)
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    m = np.column_stack((x, y, z))
    # Rotation matrix -> quaternion (xyzw), branching on the largest diagonal term.
    t = np.trace(m)
    if t > 0:
        s = 2 * np.sqrt(1 + t)
        q = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, s / 4]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2 * np.sqrt(1 + m[0, 0] - m[1, 1] - m[2, 2])
        q = [s / 4, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s]
    elif m[1, 1] > m[2, 2]:
        s = 2 * np.sqrt(1 + m[1, 1] - m[0, 0] - m[2, 2])
        q = [(m[0, 1] + m[1, 0]) / s, s / 4, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s]
    else:
        s = 2 * np.sqrt(1 + m[2, 2] - m[0, 0] - m[1, 1])
        q = [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, s / 4, (m[1, 0] - m[0, 1]) / s]
    return np.array(q)


def rotate_z(q):
    """The +Z axis of the rotation q (xyzw)."""
    x, y, z, w = q
    return np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])


def _frame_publisher(node):
    pub = node.create_publisher(TransformStamped, "/interactive_frame/set", 10)
    deadline = time.time() + 5.0
    while pub.get_subscription_count() == 0 and time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    if pub.get_subscription_count() == 0:
        sys.exit("frame_bridge_node is not subscribed to /interactive_frame/set")
    return pub


def publish(node, distance):
    views = []
    for name, look_at, direction in VIEWS:
        look_at = np.array(look_at, dtype=float)
        direction = np.array(direction, dtype=float)
        views.append((name, look_at + distance * direction / np.linalg.norm(direction), look_at))
    _publish_views(node, views)


def load(node, experiment):
    """Publish an experiment YAML's camera targets (position, look_at) as frames."""
    import yaml
    with open(experiment) as handle:
        config = yaml.safe_load(handle)
    if config.get("targets_frame", FRAME) != FRAME:
        sys.exit(f"targets_frame is {config.get('targets_frame')}, expected {FRAME}")
    views = [(t["name"], np.array(t["position"], dtype=float), np.array(t["look_at"], dtype=float))
             for t in config.get("targets") or [] if "position" in t and "look_at" in t]
    _publish_views(node, views)


def _publish_views(node, views):
    pub = _frame_publisher(node)
    for name, camera, look_at in views:
        q = optical_quaternion(look_at - camera)
        msg = TransformStamped()
        msg.header.frame_id = FRAME
        msg.child_frame_id = PREFIX + name
        msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z = camera
        (msg.transform.rotation.x, msg.transform.rotation.y,
         msg.transform.rotation.z, msg.transform.rotation.w) = q
        pub.publish(msg)
        rclpy.spin_once(node, timeout_sec=0.05)
        print(f"{msg.child_frame_id}: camera {np.round(camera, 3).tolist()} -> {np.round(look_at, 3).tolist()}")
    time.sleep(0.5)


def export(node, distance, d_min, d_max):
    state = []
    qos = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
    node.create_subscription(String, "/interactive_frames/state",
                             lambda m: state.append(json.loads(m.data)), qos)
    deadline = time.time() + 5.0
    while not state and time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    if not state:
        sys.exit("no state from frame_bridge_node on /interactive_frames/state")
    surface_points = {PREFIX + name: np.array(look_at, dtype=float) for name, look_at, _ in VIEWS}
    print("targets:")
    for f in state[-1]:
        if not f["name"].startswith(PREFIX):
            continue
        if f["parent"] != FRAME:
            print(f"  # {f['name']}: parent is {f['parent']}, not {FRAME}; skipped")
            continue
        p = np.array([f["translation"][k] for k in "xyz"])
        axis = rotate_z(np.array([f["rotation"][k] for k in "xyzw"]))
        notes = []
        if f["name"] in surface_points:
            # Generated view: keep its point on the machine; the distance/aim the
            # corrected camera has to it is what the 0.5-1.0 m range constrains.
            look_at = surface_points[f["name"]]
            d = float(np.linalg.norm(look_at - p))
            aim = np.degrees(np.arccos(np.clip(np.dot(axis, (look_at - p) / d), -1, 1)))
            if aim > 10:
                # Re-aimed in the editor: follow the new optical axis instead.
                look_at, d = p + distance * axis, distance
                notes.append(f"re-aimed {aim:.0f} deg: look_at on the new axis at {distance} m")
        else:
            look_at, d = p + distance * axis, distance
        if not d_min <= d <= d_max:
            notes.append(f"distance {d:.2f} m outside [{d_min}, {d_max}]")
        note = f"  # {'; '.join(notes)}" if notes else ""
        print(f"  - {{name: {f['name'][len(PREFIX):]}, frame: camera, "
              f"position: {np.round(p, 3).tolist()}, look_at: {np.round(look_at, 3).tolist()}}}"
              f"  # {d:.2f} m{note.replace('  #', ';') if note else ''}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["publish", "export", "load"])
    parser.add_argument("--experiment", help="load: experiment YAML whose targets to publish")
    parser.add_argument("--distance", type=float, default=0.75, help="camera to look_at (m)")
    parser.add_argument("--min", type=float, default=0.5)
    parser.add_argument("--max", type=float, default=1.0)
    args = parser.parse_args()
    if not args.min <= args.distance <= args.max:
        sys.exit(f"--distance {args.distance} outside [{args.min}, {args.max}]")
    rclpy.init()
    node = rclpy.create_node("campetella_scan_viewpoints")
    try:
        if args.mode == "publish":
            publish(node, args.distance)
        elif args.mode == "load":
            if not args.experiment:
                sys.exit("load needs --experiment <path to the experiment YAML>")
            load(node, args.experiment)
        else:
            export(node, args.distance, args.min, args.max)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
