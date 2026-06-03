// IT: Entry point React: monta il componente QuantDashboard nel div #root.
// EN: React entry point: mounts the QuantDashboard component into the #root div.
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import QuantDashboard from "./quant_dashboard_full.jsx";

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <QuantDashboard />
  </StrictMode>
);
