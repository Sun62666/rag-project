import axios from 'axios'
import { ElMessage } from 'element-plus'

const api = axios.create({
  baseURL: '',
  timeout: 120000,
})

// 请求拦截器
api.interceptors.request.use(config => {
  const token = localStorage.getItem('smartops_token')
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

// 响应拦截器 - 统一错误处理
api.interceptors.response.use(
  response => response,
  error => {
    if (!error.response) {
      // 网络错误 / 超时
      ElMessage.error('网络连接失败，请检查网络或服务是否启动')
      return Promise.reject(error)
    }
    const { status, data } = error.response
    switch (status) {
      case 401:
        localStorage.removeItem('smartops_token')
        // 避免在登录页重复跳转
        if (window.location.pathname !== '/login') {
          ElMessage.warning('登录已过期，请重新登录')
          window.location.href = '/login'
        }
        break
      case 403:
        ElMessage.error('没有权限执行此操作')
        break
      case 404:
        ElMessage.error('请求的资源不存在')
        break
      case 422:
        // 表单验证错误，由调用方处理
        break
      case 500:
        ElMessage.error('服务器内部错误，请稍后重试')
        break
      default:
        ElMessage.error(data?.detail || `请求失败 (${status})`)
    }
    return Promise.reject(error)
  }
)

// 认证 API
export const authAPI = {
  login: (data) => api.post('/auth/login', data),
  register: (data) => api.post('/auth/register', data),
  logout: () => api.post('/auth/logout'),
  me: () => api.get('/auth/me'),
}

// 会话 API
export const sessionAPI = {
  create: () => api.post('/new_session'),
  list: () => api.get('/sessions'),
  get: (id) => api.get(`/sessions/${id}`),
  delete: (id) => api.delete(`/sessions/${id}`),
  rename: (id, title) => api.put(`/sessions/${id}/rename`, { title }),
  clearHistory: (sessionId) => api.post(`/clear_history?session_id=${sessionId}`),
}

// 对话 API (SSE 流式)
export const chatAPI = {
  ask: (query, sessionId) => {
    const token = localStorage.getItem('smartops_token')
    return fetch('/ask', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
      body: JSON.stringify({ query, session_id: sessionId }),
    })
  },
}

// 模式 API
export const modeAPI = {
  get: () => api.get('/mode'),
  switch: (useAgent) => api.post('/mode', { use_agent: useAgent }),
}

// 运维工具 API
export const opsAPI = {
  // 日志
  getLogs: (params) => api.get('/ops/logs', { params }),
  getLogFiles: () => api.get('/ops/logs/files'),
  // 文档上传
  uploadDocument: (formData) => api.post('/ops/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  }),
  listDocs: () => api.get('/ops/upload/list'),
  deleteDoc: (filename) => api.delete(`/ops/upload/${encodeURIComponent(filename)}`),
  // 知识库统计
  knowledgeStats: () => api.get('/ops/knowledge/stats'),
  // 知识图谱
  kgStats: () => api.get('/ops/knowledge/graph/stats'),
  kgVis: (entity, depth) => api.get('/ops/knowledge/graph/vis', { params: { entity, depth } }),
  kgExtract: (text, method) => api.post(`/ops/knowledge/graph/extract?text=${encodeURIComponent(text)}&method=${method}`),
}

// 评估 API
export const evaluateAPI = {
  run: (threshold = 0.7) => api.post('/evaluate', { threshold }),
  getQuestions: () => api.get('/evaluate/questions'),
}

export default api
