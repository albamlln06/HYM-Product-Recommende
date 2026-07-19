import { useMemo } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { MlflowRun } from "../api";
import { FAMILY_COLORS, FAMILY_ORDER } from "../families";

interface GroupRow {
  data_config: string;
  label: string;
  n_customers: number;
  scores: Partial<Record<string, number>>;
}

function buildGroups(runs: MlflowRun[]): GroupRow[] {
  const groups = new Map<string, GroupRow>();

  for (const r of runs) {
    const configId = r.params.data_config;
    if (!configId || r.status !== "FINISHED" || r.map12 === null) continue;

    if (!groups.has(configId)) {
      const nCustomers = Number(r.params.n_customers ?? 0);
      const minPurchases = r.params.min_purchases ?? "?";
      const maxMonths = r.params.max_months_since_last_purchase ?? "?";
      groups.set(configId, {
        data_config: configId,
        label: `${nCustomers.toLocaleString("es-ES")} clientes (min ${minPurchases} compras, ${maxMonths}m)`,
        n_customers: nCustomers,
        scores: {},
      });
    }

    const row = groups.get(configId)!;
    const current = row.scores[r.family];
    if (current === undefined || r.map12 > current) {
      row.scores[r.family] = r.map12;
    }
  }

  return [...groups.values()].sort((a, b) => a.n_customers - b.n_customers);
}

function winnerFamily(row: GroupRow): string | null {
  let best: string | null = null;
  let bestVal = -Infinity;
  for (const family of FAMILY_ORDER) {
    const v = row.scores[family];
    if (v !== undefined && v > bestVal) {
      bestVal = v;
      best = family;
    }
  }
  return best;
}

function formatScore(v: number | undefined): string {
  return v === undefined ? "—" : v.toFixed(4);
}

export default function ComparisonPanel({ runs }: { runs: MlflowRun[] }) {
  const groups = useMemo(() => buildGroups(runs), [runs]);
  const familiesPresent = FAMILY_ORDER.filter((f) => groups.some((g) => g.scores[f] !== undefined));

  const chartData = groups.map((g) => ({ label: g.label, ...g.scores }));

  if (groups.length === 0) {
    return (
      <p className="muted">
        Todavía ningún run tiene la etiqueta de configuración de datos necesaria para
        compararlos (los runs previos a este cambio no la llevan). Vuelve a lanzar
        <code> train.py</code> o <code>experiment_xgboost.py</code> para que aparezcan aquí.
      </p>
    );
  }

  return (
    <div style={{ marginBottom: 8 }}>
      <h3 style={{ marginTop: 0 }}>Comparativa por configuración de datos</h3>
      <p className="muted">
        Mejor MAP@12 de cada modelo para cada muestra de clientes (mismo split de
        evaluación dentro de cada configuración).
      </p>

      <div className="chart-card">
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={chartData} margin={{ top: 16, right: 16, left: 8, bottom: 32 }}>
            <CartesianGrid vertical={false} stroke="var(--gridline)" />
            <XAxis
              dataKey="label"
              tick={{ fill: "var(--text-muted)", fontSize: 11 }}
              tickLine={false}
              axisLine={{ stroke: "var(--axis)" }}
              interval={0}
              angle={-12}
              textAnchor="end"
              height={56}
            />
            <YAxis
              tickFormatter={(v: number) => v.toFixed(4)}
              tick={{ fill: "var(--text-muted)", fontSize: 12 }}
              tickLine={false}
              axisLine={false}
              width={64}
            />
            <Tooltip formatter={(v: unknown) => (typeof v === "number" ? v.toFixed(4) : String(v))} />
            <Legend />
            {familiesPresent.map((family) => (
              <Bar key={family} dataKey={family} name={family} fill={FAMILY_COLORS[family]} radius={[3, 3, 0, 0]} />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>

      <table className="metrics-table" style={{ maxWidth: 640 }}>
        <thead>
          <tr>
            <th>Configuración</th>
            {familiesPresent.map((f) => (
              <th key={f}>{f}</th>
            ))}
            <th>Gana</th>
          </tr>
        </thead>
        <tbody>
          {groups.map((g) => {
            const win = winnerFamily(g);
            return (
              <tr key={g.data_config}>
                <td>{g.label}</td>
                {familiesPresent.map((f) => (
                  <td key={f} className="tabular" style={{ fontWeight: f === win ? 600 : 400 }}>
                    {formatScore(g.scores[f])}
                  </td>
                ))}
                <td>{win ?? "—"}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
