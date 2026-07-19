import { useEffect, useMemo, useState } from "react";
import {
  CartesianGrid,
  Legend,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ResponsiveContainer,
} from "recharts";
import { getMlflowRuns, type MlflowRun } from "../api";
import { FAMILY_COLORS, FAMILY_ORDER } from "../families";
import ComparisonPanel from "./ComparisonPanel";

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
  payload: MlflowRun & { idx: number };
}

function RunTooltip({ active, payload }: { active?: boolean; payload?: TooltipPayloadItem[] }) {
  if (!active || !payload || payload.length === 0) return null;
  const run = payload[0].payload;
  return (
    <div className="chart-card" style={{ margin: 0, padding: 10, maxWidth: 320 }}>
      <strong>{run.run_name}</strong>
      <div className="muted">{run.family} · {run.status}</div>
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
    () => runs.map((r, i) => ({ ...r, idx: i + 1 })),
    [runs]
  );

  const familiesPresent = FAMILY_ORDER.filter((f) => runs.some((r) => r.family === f));

  const runsDesc = useMemo(
    () => [...runs].reverse(),
    [runs]
  );

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

      {runs.length > 0 && <ComparisonPanel runs={runs} />}

      {runs.length > 0 && (
        <>
          <h3>Historial completo de runs</h3>
          <div className="legend">
            {familiesPresent.map((f) => (
              <span className="legend-item" key={f}>
                <span className="legend-dot" style={{ background: FAMILY_COLORS[f] }} />
                {f}
              </span>
            ))}
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
                <Legend />
                {familiesPresent.map((family) => (
                  <Scatter
                    key={family}
                    name={family}
                    data={chartData.filter((r) => r.family === family)}
                    fill={FAMILY_COLORS[family]}
                  />
                ))}
              </ScatterChart>
            </ResponsiveContainer>
          </div>

          <table className="runs-table">
            <thead>
              <tr>
                <th>Run</th>
                <th>Familia</th>
                <th>Estado</th>
                <th>Fecha</th>
                <th>MAP@12</th>
                <th>Parámetros</th>
              </tr>
            </thead>
            <tbody>
              {runsDesc.map((r) => (
                <tr key={r.run_id}>
                  <td>{r.run_name}</td>
                  <td>{r.family}</td>
                  <td><span className={`status-badge ${r.status}`}>{r.status}</span></td>
                  <td className="muted">{formatTimestamp(r.start_time)}</td>
                  <td className="tabular">{r.map12 !== null ? r.map12.toFixed(4) : "—"}</td>
                  <td className="params-cell">{formatParams(r.params)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </section>
  );
}
