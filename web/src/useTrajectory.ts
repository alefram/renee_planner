import { useCallback, useEffect, useState } from "react";
import { getCoverage, getTrajectory } from "./api";
import type { Coverage, Trajectory } from "./types";

/** Loads a trajectory and its coverage; `reload` refetches (e.g. on a live /scan_trajectory). */
export function useTrajectory(name: string | null) {
  const [trajectory, setTrajectory] = useState<Trajectory | null>(null);
  const [coverage, setCoverage] = useState<Coverage | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [version, setVersion] = useState(0);

  useEffect(() => {
    if (!name) {
      setTrajectory(null);
      setCoverage(null);
      return;
    }
    let cancelled = false;
    setLoading(true);
    setError(null);
    Promise.all([getTrajectory(name), getCoverage(name)])
      .then(([t, c]) => {
        if (cancelled) return;
        setTrajectory(t);
        setCoverage(c.points ? (c as Coverage) : null);
      })
      .catch((e: Error) => !cancelled && setError(e.message))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [name, version]);

  const reload = useCallback(() => setVersion((v) => v + 1), []);
  return { trajectory, coverage, error, loading, reload };
}
