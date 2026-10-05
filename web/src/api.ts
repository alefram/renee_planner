// HTTP access to scan_viewer_server_node (same origin; `npm run dev` proxies /api).
import type { Coverage, Trajectory, TrajectorySummary } from "./types";

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(path, { cache: "no-store" });
  const body = await response.json();
  if (!response.ok) throw new Error(body?.error ?? `${response.status} ${response.statusText}`);
  return body as T;
}

export const listTrajectories = () => getJson<TrajectorySummary[]>("api/trajectories");
export const getTrajectory = (name: string) => getJson<Trajectory>(`api/trajectories/${encodeURIComponent(name)}`);
export const getCoverage = (name: string) =>
  getJson<Partial<Coverage>>(`api/trajectories/${encodeURIComponent(name)}/coverage`);
export const getConfig = () => getJson<{ rosbridge_port: number }>("api/config");

export const meshUrl = (trajectory: Trajectory) =>
  `api/meshes/${encodeURIComponent(trajectory.machine.mesh.replace(/\.glb$/, ""))}.glb`;
export const yamlUrl = (name: string) => `api/trajectories/${encodeURIComponent(name)}.yaml`;
