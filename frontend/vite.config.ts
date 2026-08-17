import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The API runs as a separate process (it owns the GPU and the Stockfish child
// process), so dev traffic is proxied rather than served from the same origin.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8077',
        changeOrigin: true,
        ws: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
  },
})
