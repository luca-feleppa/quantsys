// IT: Config Vite: plugin React + dev server sulla porta 5173 (apre il browser).
// EN: Vite config: React plugin + dev server on port 5173 (auto-opens the browser).
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: { port: 5173, open: true },
});
