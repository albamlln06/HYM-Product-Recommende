import { useState } from "react";
import CustomerPanel from "./CustomerPanel";
import MetricsPanel from "./MetricsPanel";
import EvolutionPanel from "./EvolutionPanel";

type Tab = "customer" | "metrics" | "evolution";

export default function AnalysisPanel({ onSwitchToStore }: { onSwitchToStore: () => void }) {
  const [tab, setTab] = useState<Tab>("customer");

  return (
    <div className="app">
      <header className="app-header">
        <h1 className="brand">
          <span className="brand-mark" aria-hidden="true">S</span>
          <span className="brand-word">
            STUDIO<span className="brand-dot">.</span>
          </span>
        </h1>
        <p className="brand-tagline">Panel de análisis de los modelos de recomendación</p>
        <div className="analysis-nav-row">
          <nav className="tabs">
            <button
              className={tab === "customer" ? "tab active" : "tab"}
              onClick={() => setTab("customer")}
            >
              Recomendaciones por cliente
            </button>
            <button
              className={tab === "metrics" ? "tab active" : "tab"}
              onClick={() => setTab("metrics")}
            >
              Comparativa de métricas
            </button>
            <button
              className={tab === "evolution" ? "tab active" : "tab"}
              onClick={() => setTab("evolution")}
            >
              Evolución de modelos
            </button>
          </nav>
          <button className="refresh-button" onClick={onSwitchToStore}>
            ← Volver a la tienda
          </button>
        </div>
      </header>

      <main>
        {tab === "customer" && <CustomerPanel />}
        {tab === "metrics" && <MetricsPanel />}
        {tab === "evolution" && <EvolutionPanel />}
      </main>
    </div>
  );
}
