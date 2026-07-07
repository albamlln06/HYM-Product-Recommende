import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  LabelList,
  ResponsiveContainer,
  XAxis,
  YAxis,
} from "recharts";
import { getMetrics, type MetricsResponse } from "../api";

function formatMap12(value: number): string {
  return value.toFixed(4);
}

export default function MetricsPanel() {
  const [data, setData] = useState<MetricsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getMetrics()
      .then(setData)
      .catch((err) => setError(err.message));
  }, []);

  if (error) return <p className="error">Error al cargar métricas: {error}</p>;
  if (!data) return <p className="muted">Cargando métricas…</p>;

  return (
    <section>
      <h2>Comparativa de modelos — MAP@{data.meta.k}</h2>
      <p className="muted">
        Evaluado con leave-one-out sobre {data.meta.n_eval_users.toLocaleString("es-ES")} clientes
        activos (muestra de {data.meta.n_customers_sampled.toLocaleString("es-ES")}).
      </p>

      <div className="chart-card">
        <ResponsiveContainer width="100%" height={320}>
          <BarChart data={data.metrics} margin={{ top: 24, right: 16, left: 8, bottom: 8 }} barCategoryGap="30%">
            <CartesianGrid vertical={false} stroke="var(--gridline)" />
            <XAxis
              dataKey="model"
              tickLine={false}
              axisLine={{ stroke: "var(--axis)" }}
              tick={{ fill: "var(--text-muted)", fontSize: 13 }}
            />
            <YAxis
              tickFormatter={formatMap12}
              tickLine={false}
              axisLine={false}
              tick={{ fill: "var(--text-muted)", fontSize: 12 }}
              width={64}
            />
            <Bar dataKey="map12" fill="var(--series-1)" radius={[4, 4, 0, 0]} maxBarSize={64}>
              <LabelList
                dataKey="map12"
                position="top"
                formatter={(value: unknown) => formatMap12(Number(value))}
                style={{ fill: "var(--text-secondary)", fontSize: 12 }}
              />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <table className="metrics-table">
        <thead>
          <tr>
            <th>Modelo</th>
            <th>MAP@{data.meta.k}</th>
          </tr>
        </thead>
        <tbody>
          {data.metrics.map((m) => (
            <tr key={m.model}>
              <td>{m.model}</td>
              <td className="tabular">{formatMap12(m.map12)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}
