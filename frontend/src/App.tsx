import { useState } from "react";
import "./App.css";
import MetricsPanel from "./components/MetricsPanel";
import CustomerPanel from "./components/CustomerPanel";

type Tab = "customer" | "metrics";

function App() {
  const [tab, setTab] = useState<Tab>("customer");

  return (
    <div className="app">
      <header className="app-header">
        <h1>Panel de recomendación de productos</h1>
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
        </nav>
      </header>

      <main>{tab === "customer" ? <CustomerPanel /> : <MetricsPanel />}</main>
    </div>
  );
}

export default App;
