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
  payload: MlflowRun & { idx: number; group: RunGroup };
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
      {Object.keys(run.params).length > 0 && (
        <div className="muted" style={{ marginTop: 4 }}>{formatParams(run.params)}</div>
      )}
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

  const chartData = useMemo(
    () => runs.map((r, i) => ({ ...r, idx: i + 1, group: runGroup(r) })),
    [runs]
  );

  // Grupos presentes (por familia, o por combinación n_customers/candidate_pool_size
  // dentro de Optuna), uno por color/entrada de leyenda. Ordenados por
  // etiqueta para que el orden en la leyenda sea estable entre refrescos.
  const groupsPresent = useMemo(() => {
    const byKey = new Map<string, RunGroup>();
    for (const r of chartData) {
      if (!byKey.has(r.group.key)) byKey.set(r.group.key, r.group);
    }
    return [...byKey.values()].sort((a, b) => a.label.localeCompare(b.label));
  }, [chartData]);

  const toggleGroup = (key: string) => {
    setHiddenGroups((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  };

  // El filtro afecta a gráfico y tabla por igual, para que ambos cuenten
  // siempre la misma historia. Se conserva el idx original (orden real de
  // ejecución) en vez de renumerar, así el eje X no cambia de significado
  // al ocultar/mostrar grupos.
  const visibleChartData = useMemo(
    () => chartData.filter((r) => !hiddenGroups.has(r.group.key)),
    [chartData, hiddenGroups]
  );

  // Mejor run (mayor MAP@12) entre los actualmente visibles, para que la
  // tarjeta coincida siempre con lo que se ve en el gráfico.
  const bestRun = useMemo(() => {
    let best: (typeof visibleChartData)[number] | null = null;
    for (const r of visibleChartData) {
      if (r.map12 === null) continue;
      if (!best || best.map12 === null || r.map12 > best.map12) best = r;
    }
    return best;
  }, [visibleChartData]);

  return (
    <section>
      <h2>Evolución de modelos (MLflow)</h2>
      <p className="muted">
        Historial de runs del experimento <code>hym-recomendator</code>, ordenados cronológicamente.
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
          <div className="legend">
            {groupsPresent.map((g) => {
              const off = hiddenGroups.has(g.key);
              return (
                <button
                  type="button"
                  key={g.key}
                  className={`legend-item${off ? " legend-item--off" : ""}`}
                  aria-pressed={!off}
                  onClick={() => toggleGroup(g.key)}
                >
                  <span className="legend-dot" style={{ background: g.color }} />
                  {g.label}
                </button>
              );
            })}
          </div>

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
                {groupsPresent
                  .filter((g) => !hiddenGroups.has(g.key))
                  .map((g) => (
                    <Scatter
                      key={g.key}
                      name={g.label}
                      data={visibleChartData.filter((r) => r.group.key === g.key)}
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
              <div className="best-run-map">MAP@12 {bestRun.map12!.toFixed(4)}</div>
              {Object.keys(bestRun.params).length > 0 && (
                <div className="muted best-run-params">{formatParams(bestRun.params)}</div>
              )}
            </div>
          )}
        </>
      )}
    </section>
  );
}
