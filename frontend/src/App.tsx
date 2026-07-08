import { useState } from "react";
import "./App.css";
import Home from "./components/Home";
import MetricsPanel from "./components/MetricsPanel";
import CustomerPanel from "./components/CustomerPanel";
import EvolutionPanel from "./components/EvolutionPanel";

type Tab = "home" | "customer" | "metrics" | "evolution";

function App() {
  const [tab, setTab] = useState<Tab>("home");

  return (
    <div className="app">
      <header className="app-header">
        <h1>Panel de recomendación de productos</h1>
        <nav className="tabs">
          <button
            className={tab === "home" ? "tab active" : "tab"}
            onClick={() => setTab("home")}
          >
            Inicio
          </button>
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
      </header>

      <main>
        {tab === "home" && <Home />}
        {tab === "customer" && <CustomerPanel />}
        {tab === "metrics" && <MetricsPanel />}
        {tab === "evolution" && <EvolutionPanel />}
      </main>
    </div>
  );
}

export default App;
