import { Component, type ReactNode, Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { getConfig, listTrajectories, meshUrl } from "./api";
import { COLORS } from "./colors";
import { Compare } from "./panel/Compare";
import { Export } from "./panel/Export";
import { Filters, type Layers } from "./panel/Filters";
import { PickedLinks } from "./panel/PickedLinks";
import { PoseDetail, PoseList } from "./panel/PoseList";
import { Stats } from "./panel/Stats";
import { connectRos, type RosStatus, type TfTree } from "./ros";
import { CameraPoses } from "./scene/CameraPoses";
import { Coverage } from "./scene/Coverage";
import { LiveCamera } from "./scene/LiveCamera";
import { Machine } from "./scene/Machine";
import { Lane, Route } from "./scene/Route";
import { Scene } from "./scene/Scene";
import type { Quat, TrajectorySummary, Vec3 } from "./types";
import { linkOf, nearestPoint, outsideTarget, type PickedLink } from "./picking";
import { useTrajectory } from "./useTrajectory";

/** Keeps a missing or broken mesh from taking the whole scene down. */
class SceneBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  render() {
    return this.state.failed ? null : this.props.children;
  }
}

const STATUS_STYLE: Record<RosStatus, string> = {
  off: "bg-slate-400",
  connecting: "bg-amber-400",
  connected: "bg-emerald-500",
  error: "bg-red-500",
};

export default function App() {
  const [items, setItems] = useState<TrajectorySummary[]>([]);
  const [current, setCurrent] = useState<string | null>(() => decodeURIComponent(location.hash.slice(1)) || null);
  const [compare, setCompare] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [layers, setLayers] = useState<Layers>({ machine: true, coverage: true, frustums: true, route: true, lane: true, labels: false, live: true });
  const [depth, setDepth] = useState(0.3);
  const [opacity, setOpacity] = useState(0.85);
  const [visibleSections, setVisibleSections] = useState<Set<number>>(new Set());
  const [rosStatus, setRosStatus] = useState<RosStatus>("off");
  const [livePose, setLivePose] = useState<{ position: Vec3; orientation: Quat } | null>(null);
  const [picks, setPicks] = useState<PickedLink[]>([]);

  const main = useTrajectory(current);
  const other = useTrajectory(compare);
  const { trajectory, coverage } = main;

  // What is on screen, read by the polling below without restarting it.
  const shownRef = useRef({ main, other, current, compare });
  shownRef.current = { main, other, current, compare };

  /** Reloads the list; reloads a shown trajectory whose file was regenerated (or every shown one with force). */
  const refresh = useCallback((force = false) => {
    listTrajectories()
      .then((list) => {
        setItems((previous) => (JSON.stringify(previous) === JSON.stringify(list) ? previous : list));
        setCurrent((name) => name ?? list.find((t) => !t.error)?.name ?? null);
        const shown = shownRef.current;
        const stamp = (name: string | null) => list.find((t) => t.name === name)?.generated;
        for (const [name, view] of [[shown.current, shown.main], [shown.compare, shown.other]] as const) {
          if (!name || view.loading) continue;
          const changed = view.trajectory && stamp(name) && stamp(name) !== view.trajectory.generated;
          if (force || changed) view.reload();
        }
      })
      .catch(() => undefined); // keep what is shown if the server is briefly away
  }, []);

  // Live without rosbridge: poll the (small) list every 3 s while the tab is visible.
  useEffect(() => {
    refresh();
    const timer = setInterval(() => document.visibilityState === "visible" && refresh(), 3000);
    return () => clearInterval(timer);
  }, [refresh]);

  useEffect(() => {
    if (current) history.replaceState(null, "", `#${encodeURIComponent(current)}`);
    setSelectedId(null);
  }, [current]);

  useEffect(() => {
    setVisibleSections(new Set(coverage?.section_names.map((_, i) => i) ?? []));
  }, [coverage]);

  // Live (rosbridge): a new /scan_trajectory refreshes at once; /tf gives where the camera is now.
  const liveRef = useRef({ frame: "robot_map", cameraFrame: "" });
  liveRef.current = { frame: trajectory?.frame ?? "robot_map", cameraFrame: trajectory?.camera.frame ?? "" };
  useEffect(() => {
    let disconnect: (() => void) | null = null;
    let lastTf = 0;
    getConfig()
      .then(({ rosbridge_port }) => {
        const protocol = location.protocol === "https:" ? "wss" : "ws";
        disconnect = connectRos(`${protocol}://${location.hostname || "localhost"}:${rosbridge_port}`, {
          onStatus: setRosStatus,
          // A new trajectory: refresh now instead of at the next poll (it reloads what changed).
          onTrajectory: () => refresh(),
          onTf: (tree: TfTree) => {
            const now = performance.now();
            if (now - lastTf < 100 || !liveRef.current.cameraFrame) return;
            lastTf = now;
            setLivePose(tree.lookup(liveRef.current.frame, liveRef.current.cameraFrame));
          },
        });
      })
      .catch(() => setRosStatus("off"));
    return () => disconnect?.();
  }, [refresh]);

  const visiblePoses = useMemo(() => {
    if (!trajectory) return [];
    const names = coverage?.section_names;
    if (!names) return trajectory.poses;
    return trajectory.poses.filter((p) => visibleSections.has(names.indexOf(p.section)) || names.indexOf(p.section) < 0);
  }, [trajectory, coverage, visibleSections]);

  const { center, extent } = useMemo(() => {
    const poses = trajectory?.poses ?? [];
    if (!poses.length) return { center: null as Vec3 | null, extent: 6 };
    const c = [0, 1, 2].map((k) => poses.reduce((s, p) => s + p.position[k], 0) / poses.length) as Vec3;
    const extent = Math.max(...poses.map((p) => Math.hypot(p.position[0] - c[0], p.position[1] - c[1], p.position[2] - c[2])));
    return { center: c, extent };
  }, [trajectory?.experiment]);

  const selected = trajectory?.poses.find((p) => p.id === selectedId) ?? null;

  // Which link is this? Log the clicked surface point's link (panel + browser console).
  const pickIndex = useCallback(
    (index: number) => {
      if (!coverage) return;
      const pick = linkOf(coverage, index);
      if (!pick) return;
      console.info(`[scan_viewer] link ${pick.link} (section ${pick.section}) at [${pick.point.map((v) => v.toFixed(3)).join(", ")}]`);
      setPicks((list) => [pick, ...list].slice(0, 12));
    },
    [coverage],
  );
  const pickPoint = useCallback(
    (point: Vec3) => {
      if (!coverage) return;
      const { index, distance } = nearestPoint(coverage.points, point);
      // The surface only holds the target's parts: a far click hit another part of the mesh.
      if (distance <= 0.05) return pickIndex(index);
      console.info(`[scan_viewer] clicked a part outside the target at [${point.map((v) => v.toFixed(3)).join(", ")}]`);
      setPicks((list) => [outsideTarget(point), ...list].slice(0, 12));
    },
    [coverage, pickIndex],
  );
  useEffect(() => {
    setPicks([]);
  }, [current]);
  const fov = trajectory?.camera.fov_deg ?? [60, 40];

  return (
    <div className="flex h-screen flex-col overflow-hidden md:flex-row">
      <aside className="order-2 h-[45vh] w-full shrink-0 overflow-y-auto border-slate-200 bg-white md:order-1 md:h-full md:w-[22rem] md:border-r dark:border-slate-800 dark:bg-slate-950">
        <header className="flex items-center justify-between px-4 pb-1 pt-3">
          <h1 className="text-base font-semibold">Scan poses</h1>
          <span className="flex items-center gap-1.5 text-xs text-slate-500" title="rosbridge">
            <span className={`h-2 w-2 rounded-full ${STATUS_STYLE[rosStatus]}`} />
            ROS {rosStatus}
          </span>
        </header>
        <Compare items={items} current={current} compare={compare} onCurrent={setCurrent} onCompare={setCompare} onRefresh={() => refresh(true)} />
        {main.error && <p className="px-4 py-2 text-sm text-red-600">{main.error}</p>}
        {main.loading && <p className="px-4 py-2 text-sm text-slate-500">Loading…</p>}
        {trajectory && (
          <>
            <Stats trajectory={trajectory} />
            {selected && <PoseDetail pose={selected} />}
            <Filters
              layers={layers}
              setLayers={setLayers}
              sections={coverage?.section_names ?? []}
              visibleSections={visibleSections}
              setVisibleSections={setVisibleSections}
              depth={depth}
              setDepth={setDepth}
              opacity={opacity}
              setOpacity={setOpacity}
              liveAvailable={rosStatus === "connected"}
            />
            <PickedLinks picks={picks} onClear={() => setPicks([])} missingParts={Boolean(coverage && !coverage.part_ids)} />
            <PoseList poses={visiblePoses} total={trajectory.poses.length} selectedId={selectedId} onSelect={setSelectedId} />
            <Export name={trajectory.experiment} selected={selected} />
          </>
        )}
        {!trajectory && !main.loading && (
          <p className="px-4 py-3 text-sm text-slate-500">
            No trajectory loaded. Generate one with <code>scan_pose_generator.launch.py</code> and press Refresh.
          </p>
        )}
      </aside>
      <main className="relative order-1 min-h-0 flex-1 md:order-2">
        <Scene center={center} extent={extent}>
          {trajectory && layers.machine && (
            <SceneBoundary key={trajectory.machine.mesh + trajectory.generated}>
              <Suspense fallback={null}>
                <Machine url={`${meshUrl(trajectory)}?v=${encodeURIComponent(trajectory.generated)}`} opacity={opacity} onPick={pickPoint} />
              </Suspense>
            </SceneBoundary>
          )}
          {coverage && layers.coverage && (
            <Coverage data={coverage} selectedPose={selectedId} visibleSections={visibleSections} size={0.02} onPick={pickIndex} />
          )}
          {coverage && layers.lane && <Lane laneXy={coverage.lane_xy} />}
          {trajectory && layers.route && <Route poses={trajectory.poses} />}
          {trajectory && layers.frustums && (
            <CameraPoses
              poses={visiblePoses}
              total={trajectory.poses.length}
              fovDeg={fov}
              depth={depth}
              selectedId={selectedId}
              onSelect={(id) => setSelectedId((s) => (s === id ? null : id))}
              showLabels={layers.labels}
            />
          )}
          {other.trajectory && compare && (
            <CameraPoses poses={other.trajectory.poses} total={other.trajectory.poses.length} fovDeg={fov} depth={depth} showLabels={false} ghostColor={COLORS.compare} />
          )}
          {livePose && layers.live && rosStatus === "connected" && <LiveCamera pose={livePose} fovDeg={fov} depth={depth} />}
        </Scene>
        {picks[0] && (
          <div className="pointer-events-none absolute bottom-3 left-3 rounded-md bg-black/60 px-2 py-1 font-mono text-xs text-white">
            {picks[0].link} · {picks[0].section}
          </div>
        )}
        {selected && (
          <div className="pointer-events-none absolute left-3 top-3 rounded-md bg-black/60 px-2 py-1 text-xs text-white">
            Pose {selected.id} · {selected.section} · sees {selected.covered} points
          </div>
        )}
      </main>
    </div>
  );
}
