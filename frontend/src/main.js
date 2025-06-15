import { createApp } from 'vue'
import App from './App.vue'
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'
import '@/assets/styles/global.css'
import '@/assets/styles/tailwind.css'

const app = createApp(App)
app.use(ElementPlus)
app.mount('#app')