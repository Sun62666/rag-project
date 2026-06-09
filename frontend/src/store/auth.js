import { defineStore } from 'pinia'
import { ref } from 'vue'
import { authAPI } from '../api'

export const useAuthStore = defineStore('auth', () => {
  const isLoggedIn = ref(false)
  const username = ref('')

  const checkAuth = async () => {
    const token = localStorage.getItem('smartops_token')
    if (!token) {
      isLoggedIn.value = false
      return
    }
    try {
      const res = await authAPI.me()
      if (res.data.username) {
        username.value = res.data.username
        isLoggedIn.value = true
      }
    } catch {
      localStorage.removeItem('smartops_token')
      isLoggedIn.value = false
    }
  }

  const login = async (user, password) => {
    const res = await authAPI.login({ username: user, password })
    localStorage.setItem('smartops_token', res.data.token)
    username.value = res.data.username
    isLoggedIn.value = true
    return res.data
  }

  const register = async (user, password) => {
    const res = await authAPI.register({ username: user, password })
    localStorage.setItem('smartops_token', res.data.token)
    username.value = res.data.username
    isLoggedIn.value = true
    return res.data
  }

  const logout = async () => {
    try {
      await authAPI.logout()
    } catch { /* ignore */ }
    localStorage.removeItem('smartops_token')
    isLoggedIn.value = false
    username.value = ''
  }

  return { isLoggedIn, username, checkAuth, login, register, logout }
})
