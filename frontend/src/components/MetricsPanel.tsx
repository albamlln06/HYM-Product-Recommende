import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  LabelList,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { getMetrics, type MetricEntry, type MetricsResponse } from "../api";
import { FAMILY_COLORS } from "../families";

function formatDecimal(value: number): string {
  return value.toFixed(4);
}

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

interface MetricChartProps {
  title: string;
  subtitle: string;
  metrics: MetricEntry[];
  dataKey: "map12" | "hit_rate" | "category_hit_rate_test";
  formatter: (value: number) => string;
}

// Cada métrica tiene su propia escala (MAP@12 vive en 0–0.05, los % de
// aciertos en 0–1): un solo gráfico con las tres mezcladas haría invisibles
// a las más pequeñas. Small multiples con eje Y propio para cada una.
function MetricChart({ title, subtitle, metrics, dataKey, formatter }: MetricChartProps) {
  return (
    <div className="chart-card metric-chart-card">
      <h4 className="metric-chart-title">{title}</h4>
      <p className="muted metric-chart-subtitle">{subtitle}</p>
      <ResponsiveContainer width="100%" height={240}>
        <BarChart data={metrics} margin={{ top: 20, right: 12, left: 4, bottom: 8 }} barCategoryGap="30%">
          <CartesianGrid vertical={false} stroke="var(--gridline)" />
          <XAxis
            dataKey="model"
            tickLine={false}
            axisLine={{ stroke: "var(--axis)" }}
            tick={{ fill: "var(--text-muted)", fontSize: 12 }}
          />
          <YAxis
            tickFormatter={(v: number) => formatter(v)}
            tickLine={false}
            axisLine={false}
            tick={{ fill: "var(--text-muted)", fontSize: 11 }}
            width={54}
          />
          <Tooltip
            cursor={{ fill: "var(--page)" }}
            formatter={(value: unknown) => formatter(Number(value))}
            contentStyle={{
              background: "var(--surface-1)",
              border: "1px solid var(--border)",
              borderRadius: "var(--radius-m)",
              fontSize: 12,
            }}
          />
          <Bar dataKey={dataKey} radius={[4, 4, 0, 0]} maxBarSize={56}>
            {metrics.map((m) => (
              <Cell key={m.model} fill={FAMILY_COLORS[m.model] ?? FAMILY_COLORS.Otro} />
            ))}
            <LabelList
              dataKey={dataKey}
              position="top"
              formatter={(value: unknown) => formatter(Number(value))}
              style={{ fill: "var(--text-secondary)", fontSize: 11 }}
            />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
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
      <h2>Comparativa de modelos</h2>
      <p className="muted">
        Evaluado con leave-one-out sobre {data.meta.n_eval_users.toLocaleString("es-ES")} clientes
        activos (muestra de {data.meta.n_customers_sampled.toLocaleString("es-ES")}). Mismo color por
        modelo que en el panel de evolución.
      </p>

      <div className="metric-grid">
        <MetricChart
          title={`MAP@${data.meta.k}`}
          subtitle="Precisión media del ranking (métrica principal)"
          metrics={data.metrics}
          dataKey="map12"
          formatter={formatDecimal}
        />
        <MetricChart
          title="% Aciertos exactos"
          subtitle="Usuarios con al menos 1 acierto exacto en sus recos"
          metrics={data.metrics}
          dataKey="hit_rate"
          formatter={formatPercent}
        />
        <MetricChart
          title="% Acierto de categoría"
          subtitle="Recos dentro de la categoría que el usuario compró"
          metrics={data.metrics}
          dataKey="category_hit_rate_test"
          formatter={formatPercent}
        />
      </div>

      <table className="metrics-table metrics-table--wide">
        <thead>
          <tr>
            <th>Modelo</th>
            <th>MAP@{data.meta.k}</th>
            <th>Aciertos totales</th>
            <th>% Aciertos</th>
            <th>% Categoría</th>
          </tr>
        </thead>
        <tbody>
          {data.metrics.map((m) => (
            <tr key={m.model}>
              <td>{m.model}</td>
              <td className="tabular">{formatDecimal(m.map12)}</td>
              <td className="tabular">{m.total_hits}</td>
              <td className="tabular">{formatPercent(m.hit_rate)}</td>
              <td className="tabular">{formatPercent(m.category_hit_rate_test)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}
