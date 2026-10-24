import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

/**
 * Strata web shell.
 *
 * The engine host (server/) owns hardware detection, the model registry and the
 * local inference API. In dev we proxy /api to it so the browser talks to one
 * origin, exactly like the packaged app does.
 */
export default defineConfig({
  plugins: [
    react(),
    {
      // KaTeX's stylesheet names every font three times (woff2, woff, ttf). The app's webviews all read woff2, so the other two are
      // dropped before Vite copies the files: ~300 KB of fonts instead of ~1.1 MB, and none of it loads until an answer has math (#3268).
      name: 'katex-woff2-only',
      enforce: 'pre',
      transform(code, id) {
        if (!id.includes('katex') || !id.split('?')[0].endsWith('.css')) return null;
        return code.replace(/,url\([^)]*\.woff\) format\("woff"\),url\([^)]*\.ttf\) format\("truetype"\)/g, '');
      },
    },
  ],
  server: {
    port: 5199,
    strictPort: true,
    proxy: {
      '/api': {
        target: process.env.STRATA_HOST ?? 'http://127.0.0.1:8791',
        changeOrigin: true,
        ws: true,
      },
    },
  },
  build: {
    target: 'es2022',
    outDir: 'dist',
    sourcemap: true,
  },
});
