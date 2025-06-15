import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import path from 'path'
import tailwind from 'tailwindcss'

export default defineConfig({
    plugins: [vue()],
    server: {
        host: true,
        port: 5173,
        proxy: {
            '/api': {
                target: 'http://summary:5000',
                changeOrigin: true
            }
        }
    },
    resolve: {
        alias: {
            '@': path.resolve(__dirname, './src')
        }
    }
})
