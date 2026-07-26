import { useEffect, useMemo, useState } from "react";
import {
  CartesianGrid,
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
type OptunaRun = GroupedRun & { trial: number };

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

// Coloca cada run en el eje X por su familia (una "columna" por grupo) en vez
// de por orden de ejecución, con un pequeño jitter horizontal dentro de la
// columna para que los puntos no se apilen exactamente uno sobre otro. Así
// se ve de un vistazo qué familia rinde mejor, en vez de tener que rastrear
// puntos dispersos a lo largo de una línea de tiempo.
function assignCategoryX<T extends { group: RunGroup }>(
  runs: T[],
  groups: RunGroup[]
): (T & { catX: number })[] {
  const indexByKey = new Map(groups.map((g, i) => [g.key, i]));
  const byGroup = new Map<string, T[]>();
  for (const r of runs) {
    if (!byGroup.has(r.group.key)) byGroup.set(r.group.key, []);
    byGroup.get(r.group.key)!.push(r);
  }

  const out: (T & { catX: number })[] = [];
  for (const [key, groupRuns] of byGroup) {
    const base = indexByKey.get(key) ?? 0;
    const n = groupRuns.length;
    const step = n > 1 ? Math.min(0.32 / (n - 1), 0.12) : 0;
    groupRuns.forEach((r, i) => {
      const jitter = n > 1 ? (i - (n - 1) / 2) * step : 0;
      out.push({ ...r, catX: base + jitter });
    });
  }
  return out;
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

  const optunaRuns: OptunaRun[] = useMemo(
    () =>
      runs
        .filter((r) => r.family === "Optuna")
        .map((r, i) => ({ ...r, idx: i + 1, group: runGroup(r), trial: Number(r.params.optuna_trial ?? i) })),
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

  // Posición en X por familia (no por orden cronológico), para que los
  // puntos de una misma familia se agrupen en una columna y se vea de un
  // vistazo cuál es el mejor. Se indexa contra TODOS los grupos (no solo los
  // visibles) para que ocultar una familia en la leyenda no desplace las
  // columnas del resto.
  const realRunsWithCatX = useMemo(
    () => assignCategoryX(visibleRealRuns, realGroupsPresent),
    [visibleRealRuns, realGroupsPresent]
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

  // Mejor trial de Optuna (mayor MAP@12) entre los actualmente visibles.
  const bestOptunaRun = useMemo(() => {
    let best: OptunaRun | null = null;
    for (const r of visibleOptunaRuns) {
      if (r.map12 === null) continue;
      if (!best || best.map12 === null || r.map12 > best.map12) best = r;
    }
    return best;
  }, [visibleOptunaRuns]);

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
            Modelos entrenados a escala completa (train.py / experiment_xgboost.py), agrupados por
            familia para comparar de un vistazo cuál rinde mejor. No incluye trials de Optuna: corren
            sobre una muestra reducida y no son comparables en la misma escala de MAP@12.
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
                      dataKey="catX"
                      name="Familia"
                      type="number"
                      domain={[-0.5, realGroupsPresent.length - 0.5]}
                      ticks={realGroupsPresent.map((_, i) => i)}
                      tickFormatter={(v: number) => realGroupsPresent[Math.round(v)]?.label ?? ""}
                      tickLine={false}
                      axisLine={{ stroke: "var(--axis)" }}
                      tick={{ fill: "var(--text-muted)", fontSize: 12 }}
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
                          data={realRunsWithCatX.filter((r) => r.group.key === g.key)}
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
                MAP@12 de cada trial individual, por estudio/configuración de búsqueda. Sirve para ver
                la dispersión y si hay tendencia de mejora — no para comparar contra los runs reales de arriba.
              </p>

              <GroupLegend groups={optunaGroupsPresent} hiddenGroups={hiddenGroups} onToggle={toggleGroup} />

              {visibleOptunaGroups.length === 0 ? (
                <p className="muted">Todos los estudios de Optuna están ocultos por el filtro de leyenda.</p>
              ) : (
                <div className="chart-card">
                  <ResponsiveContainer width="100%" height={320}>
                    <ScatterChart margin={{ top: 16, right: 16, left: 8, bottom: 8 }}>
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
                        dataKey="map12"
                        name="MAP@12"
                        tickFormatter={(v: number) => v.toFixed(4)}
                        tickLine={false}
                        axisLine={false}
                        tick={{ fill: "var(--text-muted)", fontSize: 12 }}
                        width={64}
                      />
                      <Tooltip content={<RunTooltip />} cursor={{ strokeDasharray: "3 3" }} />
                      {visibleOptunaGroups.map((g) => (
                        <Scatter
                          key={g.key}
                          name={g.label}
                          data={visibleOptunaRuns.filter((r) => r.group.key === g.key && r.map12 !== null)}
                          fill={g.color}
                        />
                      ))}
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>
              )}

              {bestOptunaRun && (
                <div className="chart-card stat-tile-card">
                  <span className="best-run-tag">Mejor trial de Optuna</span>
                  <div className="stat-tile-grid">
                    <div className="stat-tile">
                      <div className="stat-tile-value">{bestOptunaRun.map12!.toFixed(4)}</div>
                      <div className="stat-tile-label">MAP@12</div>
                      <div className="stat-tile-desc">
                        {bestRun && bestRun.map12
                          ? `${bestOptunaRun.map12! >= bestRun.map12 ? "+" : ""}${(((bestOptunaRun.map12! - bestRun.map12) / bestRun.map12) * 100).toFixed(0)}% vs. mejor run real`
                          : "Sobre la muestra de búsqueda, no comparable con runs reales"}
                      </div>
                    </div>
                    <div className="stat-tile">
                      <div className="stat-tile-value">{bestOptunaRun.total_hits ?? "—"}</div>
                      <div className="stat-tile-label">Aciertos totales</div>
                      <div className="stat-tile-desc">Artículos exactos acertados en el hold-out</div>
                    </div>
                    <div className="stat-tile">
                      <div className="stat-tile-value">
                        {bestOptunaRun.hit_rate !== null ? `${(bestOptunaRun.hit_rate * 100).toFixed(1)}%` : "—"}
                      </div>
                      <div className="stat-tile-label">% Aciertos</div>
                      <div className="stat-tile-desc">Usuarios con al menos 1 acierto exacto</div>
                    </div>
                  </div>
                  <div className="muted stat-tile-footer">
                    {bestOptunaRun.run_name} · {bestOptunaRun.group.label} · {formatTimestamp(bestOptunaRun.start_time)}
                  </div>
                </div>
              )}
            </>
          )}
        </>
      )}
    </section>
  );
}
