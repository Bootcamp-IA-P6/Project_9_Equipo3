import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/predict": "http://127.0.0.1:8000",
      "/predict-batch": "http://127.0.0.1:8000",
      "/predict-video": "http://127.0.0.1:8000",
      "/models": "http://127.0.0.1:8000",
      "/model-info": "http://127.0.0.1:8000",
      "/model": "http://127.0.0.1:8000",
      "/videos": "http://127.0.0.1:8000",
    },
  },
});
