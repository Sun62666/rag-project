import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  server: {
    port: 3000,
    proxy: {
      '/auth': 'http://localhost:8347',
      '/sessions': 'http://localhost:8347',
      '/new_session': 'http://localhost:8347',
      '/clear_history': 'http://localhost:8347',
      '/ask': 'http://localhost:8347',
      '/mode': 'http://localhost:8347',
      '/ops': 'http://localhost:8347',
      '/evaluate': 'http://localhost:8347',
    }
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          'vue-vendor': ['vue', 'vue-router', 'pinia'],
          'element-plus': ['element-plus', '@element-plus/icons-vue'],
          'markdown': ['marked', 'dompurify', 'highlight.js'],
        }
      }
    }
  }
})
