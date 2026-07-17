import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  // Target for the dev proxy so the frontend can call `/api/*` during local
  // development without CORS friction. Falls back to VITE_API_BASE_URL, then
  // to the conventional FastAPI dev port.
  const proxyTarget =
    env.VITE_DEV_PROXY_TARGET || env.VITE_API_BASE_URL || 'http://localhost:8000';

  return {
    plugins: [react()],
    server: {
      port: 5173,
      proxy: {
        '/api': {
          target: proxyTarget,
          changeOrigin: true,
          secure: false,
        },
      },
    },
    build: {
      outDir: 'dist',
      sourcemap: false,
      chunkSizeWarningLimit: 1200,
    },
  };
});
