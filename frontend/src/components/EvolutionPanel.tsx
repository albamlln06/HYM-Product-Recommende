import { useEffect, useMemo, useState } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ResponsiveContainer,
} from "recharts";
import { getMlflowRuns, type MlflowRun } from "../api";
import { runGroup, type RunGroup } from "../families";

const REFRESH_MS = 10_000;

type GroupedRun = MlflowRun & { idx: number; group: RunGroup };

function formatTimestamp(iso: string | null): string {
  if (!iso) return "—";
  return new Date(iso).toLocaleString("es-ES", {
    dateStyle: "short",
    timeStyle: "medium",
  });
}

function formatParams(params: Record<string, string>): string {
  return Object.entries(params)
    .map(([k, v]) => `${k}=${v}`)
    .join(" · ");
}

interface TooltipPayloadItem {
  payload: GroupedRun;
}

function RunTooltip({ active, payload }: { active?: boolean; payload?: TooltipPayloadItem[] }) {
  if (!active || !payload || payload.length === 0) return null;
  const run = payload[0].payload;
  return (
    <div className="chart-card" style={{ margin: 0, padding: 10, maxWidth: 320 }}>
      <strong>{run.run_name}</strong>
      <div className="muted">{run.group.label} · {run.status}</div>
      <div className="muted">{formatTimestamp(run.start_time)}</div>
      <div className="tabular">MAP@12: {run.map12 !== null ? run.map12.toFixed(4) : "—"}</div>
      <div className="tabular">
        Aciertos: {run.total_hits ?? "—"} · {run.hit_rate !== null ? `${(run.hit_rate * 100).toFixed(1)}%` : "—"}
      </div>
      {Object.keys(run.params).length > 0 && (
        <div className="muted" style={{ marginTop: 4 }}>{formatParams(run.params)}</div>
      )}
    </div>
  );
}

// Fila ancha {trial, [groupKey]: mejor MAP@12 acumulado hasta ese trial} para
// poder dibujar una línea por estudio de Optuna en un único eje X compartido
// (el patrón "wide format" es el que mejor soporta recharts para varias
// líneas con tooltip/crosshair compartido).
function buildOptunaProgressRows(runs: GroupedRun[], groups: RunGroup[]): Record<string, number | null>[] {
  const runsByGroup = new Map<string, GroupedRun[]>();
  for (const r of runs) {
    if (!runsByGroup.has(r.group.key)) runsByGroup.set(r.group.key, []);
    runsByGroup.get(r.group.key)!.push(r);
  }

  const bestAtTrial = new Map<string, Map<number, number>>();
  let maxTrial = 0;
  for (const g of groups) {
    const sorted = (runsByGroup.get(g.key) ?? [])
      .filter((r) => r.map12 !== null)
      .sort((a, b) => Number(a.params.optuna_trial ?? 0) - Number(b.params.optuna_trial ?? 0));
    const trialMap = new Map<number, number>();
    let best = -Infinity;
    for (const r of sorted) {
      const trial = Number(r.params.optuna_trial ?? 0);
      best = Math.max(best, r.map12 as number);
      trialMap.set(trial, best);
      maxTrial = Math.max(maxTrial, trial);
    }
    bestAtTrial.set(g.key, trialMap);
  }

  const rows: Record<string, number | null>[] = [];
  const lastKnown = new Map<string, number | null>(groups.map((g) => [g.key, null]));
  for (let trial = 0; trial <= maxTrial; trial++) {
    const row: Record<string, number | null> = { trial };
    for (const g of groups) {
      const trialMap = bestAtTrial.get(g.key)!;
      if (trialMap.has(trial)) lastKnown.set(g.key, trialMap.get(trial)!);
      row[g.key] = lastKnown.get(g.key) ?? null;
    }
    rows.push(row);
  }
  return rows;
}

interface OptunaTooltipPayloadItem {
  dataKey: string;
  value: number | null;
  color: string;
  name: string;
}

function OptunaTooltip({
  active,
  payload,
  label,
}: {
  active?: boolean;
  payload?: OptunaTooltipPayloadItem[];
  label?: number;
}) {
  if (!active || !payload || payload.length === 0) return null;
  const entries = payload.filter((p) => p.value !== null && p.value !== undefined);
  if (entries.length === 0) return null;
  return (
    <div className="chart-card" style={{ margin: 0, padding: 10, maxWidth: 280 }}>
      <strong>Trial #{label}</strong>
      {entries.map((p) => (
        <div key={p.dataKey} className="tabular" style={{ color: p.color }}>
          {p.name}: {p.value!.toFixed(4)}
        </div>
      ))}
    </div>
  );
}

function GroupLegend({
  groups,
  hiddenGroups,
  onToggle,
}: {
  groups: RunGroup[];
  hiddenGroups: Set<string>;
  onToggle: (key: string) => void;
}) {
  return (
    <div className="legend">
      {groups.map((g) => {
        const off = hiddenGroups.has(g.key);
        return (
          <button
            type="button"
            key={g.key}
            className={`legend-item${off ? " legend-item--off" : ""}`}
            aria-pressed={!off}
            onClick={() => onToggle(g.key)}
          >
            <span className="legend-dot" style={{ background: g.color }} />
            {g.label}
          </button>
        );
      })}
    </div>
  );
}

export default function EvolutionPanel() {
  const [runs, setRuns] = useState<MlflowRun[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [hiddenGroups, setHiddenGroups] = useState<Set<string>>(new Set());

  const load = () => {
    getMlflowRuns()
      .then((data) => {
        setRuns(data.runs);
        setError(null);
        setLastUpdated(new Date());
      })
      .catch((err) => setError(err.message));
  };

  useEffect(() => {
    load();
    const handle = setInterval(load, REFRESH_MS);
    return () => clearInterval(handle);
  }, []);

  const toggleGroup = (key: string) => {
    setHiddenGroups((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  };

  // Los trials de Optuna corren sobre una muestra reducida y su propia doc
  // dice explícitamente que su MAP@12 NO es comparable al de un run real de
  // train.py/experiment_xgboost.py. Mezclarlos en el mismo gráfico confundía
  // más que ayudaba, así que van en dos vistas separadas: evolución de runs
  // reales (abajo) y progreso de la búsqueda de hiperparámetros (más abajo).
  const realRuns: GroupedRun[] = useMemo(() => {
    const reals = runs.filter((r) => r.family !== "Optuna");
    return reals.map((r, i) => ({ ...r, idx: i + 1, group: runGroup(r) }));
  }, [runs]);

  const optunaRuns: GroupedRun[] = useMemo(
    () => runs.filter((r) => r.family === "Optuna").map((r, i) => ({ ...r, idx: i + 1, group: runGroup(r) })),
    [runs]
  );

  const realGroupsPresent = useMemo(() => {
    const byKey = new Map<string, RunGroup>();
    for (const r of realRuns) if (!byKey.has(r.group.key)) byKey.set(r.group.key, r.group);
    return [...byKey.values()].sort((a, b) => a.label.localeCompare(b.label));
  }, [realRuns]);

  const optunaGroupsPresent = useMemo(() => {
    const byKey = new Map<string, RunGroup>();
    for (const r of optunaRuns) if (!byKey.has(r.group.key)) byKey.set(r.group.key, r.group);
    return [...byKey.values()].sort((a, b) => a.label.localeCompare(b.label));
  }, [optunaRuns]);

  const visibleRealRuns = useMemo(
    () => realRuns.filter((r) => !hiddenGroups.has(r.group.key)),
    [realRuns, hiddenGroups]
  );

  const visibleOptunaRuns = useMemo(
    () => optunaRuns.filter((r) => !hiddenGroups.has(r.group.key)),
    [optunaRuns, hiddenGroups]
  );

  const visibleOptunaGroups = useMemo(
    () => optunaGroupsPresent.filter((g) => !hiddenGroups.has(g.key)),
    [optunaGroupsPresent, hiddenGroups]
  );

  // Mejor run REAL (mayor MAP@12) entre los actualmente visibles: se excluyen
  // a propósito los trials de Optuna, que no son comparables en escala.
  const bestRun = useMemo(() => {
    let best: GroupedRun | null = null;
    for (const r of visibleRealRuns) {
      if (r.map12 === null) continue;
      if (!best || best.map12 === null || r.map12 > best.map12) best = r;
    }
    return best;
  }, [visibleRealRuns]);

  const optunaProgressRows = useMemo(
    () => buildOptunaProgressRows(visibleOptunaRuns, visibleOptunaGroups),
    [visibleOptunaRuns, visibleOptunaGroups]
  );

  return (
    <section>
      <h2>Evolución de modelos (MLflow)</h2>
      <p className="muted">
        Historial de runs del experimento <code>hym-recomendator</code>.
      </p>

      <div className="panel-toolbar">
        <button className="refresh-button" onClick={load}>Actualizar</button>
        {lastUpdated && (
          <span className="muted">Actualizado a las {lastUpdated.toLocaleTimeString("es-ES")}</span>
        )}
      </div>

      {error && <p className="error">Error al cargar runs de MLflow: {error}</p>}

      {runs.length === 0 && !error && (
        <p className="muted">Aún no hay runs registrados en MLflow.</p>
      )}

      {runs.length > 0 && (
        <>
          <h3>Runs reales</h3>
          <p className="muted">
            Evolución cronológica de los modelos entrenados a escala completa (train.py /
            experiment_xgboost.py). No incluye trials de Optuna: corren sobre una muestra reducida y
            no son comparables en la misma escala de MAP@12.
          </p>

          {realGroupsPresent.length === 0 ? (
            <p className="muted">Aún no hay runs reales registrados.</p>
          ) : (
            <>
              <GroupLegend groups={realGroupsPresent} hiddenGroups={hiddenGroups} onToggle={toggleGroup} />

              <div className="chart-card">
                <ResponsiveContainer width="100%" height={320}>
                  <ScatterChart margin={{ top: 16, right: 16, left: 8, bottom: 8 }}>
                    <CartesianGrid stroke="var(--gridline)" />
                    <XAxis
                      dataKey="idx"
                      name="Run"
                      tickLine={false}
                      axisLine={{ stroke: "var(--axis)" }}
                      tick={{ fill: "var(--text-muted)", fontSize: 12 }}
                      label={{ value: "Orden de ejecución", position: "insideBottom", offset: -4, fill: "var(--text-muted)", fontSize: 12 }}
                    />
                    <YAxis
                      dataKey="map12"
                      name="MAP@12"
                      tickFormatter={(v: number) => v.toFixed(4)}
                      tickLine={false}
                      axisLine={false}
                      tick={{ fill: "var(--text-muted)", fontSize: 12 }}
                      width={64}
                    />
                    <Tooltip content={<RunTooltip />} cursor={{ strokeDasharray: "3 3" }} />
                    {realGroupsPresent
                      .filter((g) => !hiddenGroups.has(g.key))
                      .map((g) => (
                        <Scatter
                          key={g.key}
                          name={g.label}
                          data={visibleRealRuns.filter((r) => r.group.key === g.key)}
                          fill={g.color}
                        />
                      ))}
                  </ScatterChart>
                </ResponsiveContainer>
              </div>

              {bestRun && (
                <div className="chart-card best-run-card">
                  <span className="best-run-tag">Mejor run</span>
                  <div className="best-run-header">
                    <span className="legend-dot" style={{ background: bestRun.group.color }} />
                    <strong>{bestRun.run_name}</strong>
                    <span className={`status-badge ${bestRun.status}`}>{bestRun.status}</span>
                  </div>
                  <div className="muted">{bestRun.group.label} · {formatTimestamp(bestRun.start_time)}</div>
                  <div className="best-run-stats">
                    <div className="best-run-stat">
                      <span className="best-run-stat-label">MAP@12</span>
                      <span className="best-run-stat-value">{bestRun.map12!.toFixed(4)}</span>
                    </div>
                    <div className="best-run-stat">
                      <span className="best-run-stat-label">Aciertos totales</span>
                      <span className="best-run-stat-value">{bestRun.total_hits ?? "—"}</span>
                    </div>
                    <div className="best-run-stat">
                      <span className="best-run-stat-label">% Aciertos</span>
                      <span className="best-run-stat-value">
                        {bestRun.hit_rate !== null ? `${(bestRun.hit_rate * 100).toFixed(1)}%` : "—"}
                      </span>
                    </div>
                  </div>
                  {Object.keys(bestRun.params).length > 0 && (
                    <div className="muted best-run-params">{formatParams(bestRun.params)}</div>
                  )}
                </div>
              )}
            </>
          )}

          {optunaGroupsPresent.length > 0 && (
            <>
              <h3>Progreso de la búsqueda de hiperparámetros (Optuna)</h3>
              <p className="muted">
                Mejor MAP@12 acumulado según avanzan los trials, por estudio/configuración de búsqueda.
                Sirve para ver si la búsqueda converge — no para comparar contra los runs reales de arriba.
              </p>

              <GroupLegend groups={optunaGroupsPresent} hiddenGroups={hiddenGroups} onToggle={toggleGroup} />

              {visibleOptunaGroups.length === 0 ? (
                <p className="muted">Todos los estudios de Optuna están ocultos por el filtro de leyenda.</p>
              ) : (
                <div className="chart-card">
                  <ResponsiveContainer width="100%" height={320}>
                    <LineChart data={optunaProgressRows} margin={{ top: 16, right: 16, left: 8, bottom: 8 }}>
                      <CartesianGrid stroke="var(--gridline)" />
                      <XAxis
                        dataKey="trial"
                        type="number"
                        name="Trial"
                        tickLine={false}
                        axisLine={{ stroke: "var(--axis)" }}
                        tick={{ fill: "var(--text-muted)", fontSize: 12 }}
                        label={{ value: "Nº de trial", position: "insideBottom", offset: -4, fill: "var(--text-muted)", fontSize: 12 }}
                        allowDecimals={false}
                      />
                      <YAxis
                        tickFormatter={(v: number) => v.toFixed(4)}
                        tickLine={false}
                        axisLine={false}
                        tick={{ fill: "var(--text-muted)", fontSize: 12 }}
                        width={64}
                        label={{ value: "Mejor MAP@12 hasta el momento", angle: -90, position: "insideLeft", fill: "var(--text-muted)", fontSize: 12 }}
                      />
                      <Tooltip content={<OptunaTooltip />} cursor={{ strokeDasharray: "3 3" }} />
                      {visibleOptunaGroups.map((g) => (
                        <Line
                          key={g.key}
                          dataKey={g.key}
                          name={g.label}
                          stroke={g.color}
                          strokeWidth={2}
                          dot={false}
                          connectNulls={false}
                          type="monotone"
                          isAnimationActive={false}
                        />
                      ))}
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              )}
            </>
          )}
        </>
      )}
    </section>
  );
}
