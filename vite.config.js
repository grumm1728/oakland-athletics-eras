import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  base: "/oakland-athletics-eras/",
  plugins: [react()],
});
