import { useState } from "react";
import "./App.css";
import Storefront from "./components/Storefront";
import AnalysisPanel from "./components/AnalysisPanel";

type Mode = "store" | "panel";

function App() {
  const [mode, setMode] = useState<Mode>("store");

  if (mode === "panel") {
    return <AnalysisPanel onSwitchToStore={() => setMode("store")} />;
  }
  return <Storefront onSwitchToPanel={() => setMode("panel")} />;
}

export default App;
