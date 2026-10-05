#!/usr/bin/env python3
"""HTTP server of the scan-pose viewer (web/), no robot interaction.

Inputs: the viewer build (web_dir = web/dist), the trajectories folder
(config/trajectories: <exp>.yaml, <exp>_coverage.npz, <exp>_machine.glb).
Outputs (HTTP, port `port`):
  /                                   the viewer (web/dist)
  /api/config                         {rosbridge_port}
  /api/trajectories                   list of the trajectories (name, mode, poses, coverage)
  /api/trajectories/<exp>             the trajectory YAML as JSON
  /api/trajectories/<exp>/coverage    surface points, views per point, points seen by each pose, lane
  /api/meshes/<exp>.glb               the machine mesh of that trajectory
  /api/trajectories/<exp>.yaml        the YAML file (download)

    ros2 launch renee_trajectory_generation scan_viewer.launch.py   # + rosbridge; open http://<host>:8091
"""
import json
import mimetypes
import os
import re
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

import numpy as np
import rclpy
from rclpy.node import Node

from renee_trajectory_generation.trajectory import load_coverage, load_trajectory

NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


def _round(array: np.ndarray, decimals: int = 4) -> list:
    return np.round(np.asarray(array, dtype=np.float64), decimals).reshape(-1).tolist()


def make_handler(web_dir: str, trajectory_dir: str, rosbridge_port: int, log):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=web_dir, **kwargs)

        def log_message(self, fmt, *args):  # quiet: one line per error only
            pass

        def _send(self, status: int, body: bytes, content_type: str):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, data, status: int = 200):
            self._send(status, json.dumps(data).encode(), "application/json")

        def _file(self, path: str, content_type: str = ""):
            if not os.path.isfile(path):
                return self._json({"error": f"not found: {os.path.basename(path)}"}, 404)
            with open(path, "rb") as handle:
                self._send(200, handle.read(), content_type or mimetypes.guess_type(path)[0]
                           or "application/octet-stream")

        def do_GET(self):
            path = unquote(urlparse(self.path).path)
            if not path.startswith("/api/"):
                # Single-page app: unknown paths serve index.html.
                if path != "/" and not os.path.isfile(os.path.join(web_dir, path.lstrip("/"))):
                    self.path = "/index.html"
                return super().do_GET()
            try:
                self._api(path[len("/api/"):].strip("/").split("/"))
            except Exception as error:
                log(f"{path}: {error}")
                self._json({"error": str(error)}, 500)

        def _api(self, parts):
            if parts == ["config"]:
                return self._json({"rosbridge_port": rosbridge_port})
            if parts == ["trajectories"]:
                return self._json(self._list())
            if len(parts) == 2 and parts[0] == "meshes" and parts[1].endswith(".glb"):
                name = parts[1][:-4]
                if NAME.match(name):
                    return self._file(os.path.join(trajectory_dir, f"{name}.glb"), "model/gltf-binary")
            if len(parts) >= 2 and parts[0] == "trajectories" and NAME.match(parts[1]):
                if parts[1].endswith(".yaml") and len(parts) == 2:
                    return self._file(os.path.join(trajectory_dir, parts[1]), "application/x-yaml")
                path = os.path.join(trajectory_dir, f"{parts[1]}.yaml")
                if not os.path.isfile(path):
                    return self._json({"error": f"no trajectory '{parts[1]}'"}, 404)
                trajectory = load_trajectory(path)
                if len(parts) == 2:
                    return self._json(trajectory.to_dict())
                if parts[2:] == ["coverage"]:
                    return self._json(self._coverage(path, trajectory))
            return self._json({"error": "unknown endpoint"}, 404)

        def _list(self):
            items = []
            if not os.path.isdir(trajectory_dir):
                return items
            for name in sorted(os.listdir(trajectory_dir)):
                if not name.endswith(".yaml"):
                    continue
                try:
                    trajectory = load_trajectory(os.path.join(trajectory_dir, name))
                except Exception as error:  # a hand-edited or foreign YAML: list it as broken
                    items.append({"name": name[:-5], "error": str(error)})
                    continue
                items.append({"name": name[:-5], "mode": trajectory.mode, "poses": len(trajectory.poses),
                              "coverage": (trajectory.coverage or {}).get("ratio"),
                              "generated": trajectory.generated})
            return items

        def _coverage(self, path, trajectory):
            arrays = load_coverage(path, trajectory)
            if not arrays:
                return {}
            return {"points": _round(arrays["points_m"]), "normals": _round(arrays["normals"], 3),
                    "section_ids": arrays["section_ids"].tolist(),
                    "section_names": [str(s) for s in arrays["section_names"]],
                    "view_count": arrays["view_count"].tolist(), "coverable": arrays["coverable"].astype(int).tolist(),
                    "pose_indptr": arrays["pose_indptr"].tolist(), "pose_indices": arrays["pose_indices"].tolist(),
                    "lane_xy": _round(arrays["lane_xy"], 3),
                    # Link of each point (trajectories generated before part ids have none).
                    **({"part_ids": arrays["part_ids"].tolist(),
                        "part_names": [str(s) for s in arrays["part_names"]]} if "part_ids" in arrays else {})}

    return Handler


class ScanViewerServerNode(Node):
    def __init__(self):
        super().__init__("scan_viewer_server_node")
        web_dir = self.declare_parameter("web_dir", "").value
        trajectory_dir = self.declare_parameter("trajectory_dir", "").value
        port = int(self.declare_parameter("port", 8091).value)
        rosbridge_port = int(self.declare_parameter("rosbridge_port", 9090).value)
        if not os.path.isfile(os.path.join(web_dir, "index.html")):
            raise RuntimeError(f"no viewer build in '{web_dir}' (cd web && npm install && npm run build)")
        handler = make_handler(web_dir, trajectory_dir, rosbridge_port, self.get_logger().warning)
        self.server = ThreadingHTTPServer(("0.0.0.0", port), handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.get_logger().info(f"viewer on http://localhost:{port} (trajectories: {trajectory_dir})")

    def destroy_node(self):
        self.server.shutdown()
        super().destroy_node()


def main():
    rclpy.init()
    node = ScanViewerServerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
