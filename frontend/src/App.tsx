import { useState } from "react";
import "./App.css";
import Home from "./components/Home";
import MetricsPanel from "./components/MetricsPanel";
import CustomerPanel from "./components/CustomerPanel";
import EvolutionPanel from "./components/EvolutionPanel";
import ProfilePanel from "./components/ProfilePanel";

type Tab = "home" | "customer" | "metrics" | "evolution" | "profile";

function App() {
  const [tab, setTab] = useState<Tab>("home");

  return (
    <div className="app">
      <header className="app-header">
        <h1 className="brand">
          <span className="brand-mark" aria-hidden="true">S</span>
          <span className="brand-word">
            STUDIO<span className="brand-dot">.</span>
          </span>
        </h1>
        <p className="brand-tagline">Panel de recomendación de productos</p>
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
          <button
            className={tab === "profile" ? "tab active" : "tab"}
            onClick={() => setTab("profile")}
          >
            Mi perfil
          </button>
        </nav>
      </header>

      <main>
        {tab === "home" && <Home />}
        {tab === "customer" && <CustomerPanel />}
        {tab === "metrics" && <MetricsPanel />}
        {tab === "evolution" && <EvolutionPanel />}
        {tab === "profile" && <ProfilePanel />}
      </main>
    </div>
  );
}

export default App;
