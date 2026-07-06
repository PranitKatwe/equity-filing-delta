import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Served at the root of the Vercel deployment; the Python API lives at /api/*.
export default defineConfig({
  plugins: [react()],
});
