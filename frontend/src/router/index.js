import { createRouter, createWebHistory } from 'vue-router'
import { useAuthStore } from '../store/auth'

const routes = [
  {
    path: '/login',
    name: 'Login',
    component: () => import('../views/Login.vue'),
  },
  {
    path: '/dashboard',
    name: 'Dashboard',
    component: () => import('../views/Dashboard.vue'),
  },
  {
    path: '/chat',
    name: 'Chat',
    component: () => import('../views/Chat.vue'),
  },
  {
    path: '/tools',
    name: 'Tools',
    component: () => import('../views/Tools.vue'),
  },
  {
    path: '/evaluate',
    name: 'Evaluate',
    component: () => import('../views/Evaluate.vue'),
  },
  {
    path: '/',
    redirect: '/chat',
  },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

router.beforeEach(async (to, from, next) => {
  const auth = useAuthStore()
  // 未登录 -> 保存目标路径后跳转登录
  if (!auth.isLoggedIn && to.name !== 'Login') {
    const token = localStorage.getItem('smartops_token')
    if (token) {
      await auth.checkAuth()
      if (auth.isLoggedIn) return next()
    }
    // 保存目标路径供登录后跳转
    if (to.fullPath !== '/') {
      localStorage.setItem('smartops_redirect', to.fullPath)
    }
    return next('/login')
  }
  // 已登录访问登录页 -> 检查是否有保存的重定向
  if (auth.isLoggedIn && to.name === 'Login') {
    const redirect = localStorage.getItem('smartops_redirect')
    if (redirect) {
      localStorage.removeItem('smartops_redirect')
      return next(redirect)
    }
    return next('/chat')
  }
  next()
})

export default router