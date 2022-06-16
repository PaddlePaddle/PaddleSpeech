import { createApp } from 'vue'
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'
import Antd from 'ant-design-vue';
import 'ant-design-vue/dist/antd.css';
import App from './App.vue'
import axios from 'axios'

const app = createApp(App)
app.config.globalProperties.$http = axios

app.use(ElementPlus).use(Antd)
app.mount('#app')
