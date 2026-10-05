// Live data through rosbridge: the generator's /scan_trajectory and the camera's
// current pose (composed from /tf and /tf_static, no tf2_web_republisher needed).
import ROSLIB from "roslib";
import type { Quat, Vec3 } from "./types";

export type RosStatus = "off" | "connecting" | "connected" | "error";

interface Edge {
  parent: string;
  t: Vec3;
  q: Quat;
}

interface TransformStamped {
  header: { frame_id: string };
  child_frame_id: string;
  transform: { translation: { x: number; y: number; z: number }; rotation: { x: number; y: number; z: number; w: number } };
}

const strip = (frame: string) => frame.replace(/^\//, "");

function rotate(q: Quat, v: Vec3): Vec3 {
  const [x, y, z, w] = q;
  const [vx, vy, vz] = v;
  // v + 2 w (q x v) + 2 q x (q x v)
  const cx = y * vz - z * vy, cy = z * vx - x * vz, cz = x * vy - y * vx;
  const ccx = y * cz - z * cy, ccy = z * cx - x * cz, ccz = x * cy - y * cx;
  return [vx + 2 * (w * cx + ccx), vy + 2 * (w * cy + ccy), vz + 2 * (w * cz + ccz)];
}

function multiply(a: Quat, b: Quat): Quat {
  const [ax, ay, az, aw] = a;
  const [bx, by, bz, bw] = b;
  return [
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
    aw * bw - ax * bx - ay * by - az * bz,
  ];
}

/** Minimal TF tree: latest transform per child frame. */
export class TfTree {
  private edges = new Map<string, Edge>();

  add(transforms: TransformStamped[]) {
    for (const tf of transforms) {
      const { translation: t, rotation: r } = tf.transform;
      this.edges.set(strip(tf.child_frame_id), { parent: strip(tf.header.frame_id), t: [t.x, t.y, t.z], q: [r.x, r.y, r.z, r.w] });
    }
  }

  /** Pose of `frame` in `fixed`, or null if they are not connected. */
  lookup(fixed: string, frame: string): { position: Vec3; orientation: Quat } | null {
    let position: Vec3 = [0, 0, 0];
    let orientation: Quat = [0, 0, 0, 1];
    let current = strip(frame);
    for (let depth = 0; current !== strip(fixed); depth++) {
      const edge = this.edges.get(current);
      if (!edge || depth > 64) return null;
      const moved = rotate(edge.q, position);
      position = [moved[0] + edge.t[0], moved[1] + edge.t[1], moved[2] + edge.t[2]];
      orientation = multiply(edge.q, orientation);
      current = edge.parent;
    }
    return { position, orientation };
  }
}

export interface RosHandlers {
  onStatus: (status: RosStatus) => void;
  onTrajectory: (experiment: string) => void;
  onTf: (tree: TfTree) => void;
}

/** Connects to rosbridge; returns a function that disconnects. */
export function connectRos(url: string, handlers: RosHandlers): () => void {
  const ros = new ROSLIB.Ros({ url });
  const tree = new TfTree();
  handlers.onStatus("connecting");
  ros.on("connection", () => handlers.onStatus("connected"));
  ros.on("error", () => handlers.onStatus("error"));
  ros.on("close", () => handlers.onStatus("off"));

  const trajectory = new ROSLIB.Topic({ ros, name: "/scan_trajectory", messageType: "renee_trajectory_generation/msg/CameraTrajectory" });
  trajectory.subscribe((message) => handlers.onTrajectory((message as { experiment: string }).experiment));
  const subscribeTf = (name: string) => {
    const topic = new ROSLIB.Topic({ ros, name, messageType: "tf2_msgs/msg/TFMessage", throttle_rate: 200 });
    topic.subscribe((message) => {
      tree.add((message as { transforms: TransformStamped[] }).transforms);
      handlers.onTf(tree);
    });
    return topic;
  };
  const topics = [trajectory, subscribeTf("/tf"), subscribeTf("/tf_static")];
  return () => {
    topics.forEach((topic) => topic.unsubscribe());
    ros.close();
  };
}
