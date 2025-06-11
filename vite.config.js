import { defineConfig } from 'vite';
import tailwindcss from "@tailwindcss/vite";
import path from 'path';

export default defineConfig({
  build: {
    outDir: './static/dist',
    emptyOutDir: true,
    rollupOptions: {
      input: {
        main: path.resolve(__dirname, 'frontend/main.js'),
        chart: path.resolve(__dirname, 'frontend/chart.js'),
      },
      output: {
        entryFileNames: `[name].js`,
        chunkFileNames: `[name].js`,
        assetFileNames: `[name].[ext]`
      }
    }
  },
  plugins: [
    tailwindcss(),
  ]
});