<template>
  <div class="login-page">
    <!-- 全屏品牌背景 -->
    <div class="login-bg">
      <div class="bg-grid"></div>
      <div class="bg-glow bg-glow-1"></div>
      <div class="bg-glow bg-glow-2"></div>
    </div>

    <!-- 品牌信息层 -->
    <div class="brand-layer">
      <div class="brand-content">
        <div class="brand-logo">
          <div class="brand-logo-icon"><i class="fa-solid fa-terminal"></i></div>
          <div class="brand-logo-text">SmartOps</div>
        </div>
        <div class="brand-tagline">智能运维助手平台</div>
        <div class="brand-desc">基于大语言模型的智能运维解决方案，自动分析系统日志、定位故障根因、提供修复建议。</div>
        <div class="brand-features">
          <div class="feature-item">
            <div class="feature-icon"><i class="fa-solid fa-brain"></i></div>
            <div class="feature-text"><strong>智能分析</strong>自动解析系统日志与指标数据</div>
          </div>
          <div class="feature-item">
            <div class="feature-icon"><i class="fa-solid fa-bolt"></i></div>
            <div class="feature-text"><strong>快速定位</strong>秒级定位故障根因与影响范围</div>
          </div>
          <div class="feature-item">
            <div class="feature-icon"><i class="fa-solid fa-wand-magic-sparkles"></i></div>
            <div class="feature-text"><strong>自动修复</strong>一键生成修复方案与执行脚本</div>
          </div>
        </div>
      </div>
    </div>

    <!-- 登录卡片浮层 -->
    <div class="login-overlay">
      <div class="login-card">
        <div class="card-header">
          <div class="card-logo">
            <span class="card-logo-icon"><i class="fa-solid fa-terminal"></i></span>
            <span class="card-logo-text">SmartOps</span>
          </div>
        </div>
        <h2>{{ loginTab === 'login' ? '欢迎回来' : '创建账户' }}</h2>
        <p class="login-subtitle">{{ loginTab === 'login' ? '登录以继续使用智能运维助手' : '注册新账户开始智能运维之旅' }}</p>
        <div class="login-tabs">
          <button :class="['tab-btn', { active: loginTab === 'login' }]" @click="switchTab('login')">登录</button>
          <button :class="['tab-btn', { active: loginTab === 'register' }]" @click="switchTab('register')">注册</button>
        </div>
        <Transition name="error-fade">
          <div v-if="authError" class="auth-error">
            <div class="error-content"><i class="fa-solid fa-circle-exclamation"></i> {{ authError }}</div>
            <button type="button" class="error-close" @click="authError = ''"><i class="fa-solid fa-xmark"></i></button>
          </div>
        </Transition>
        <form @submit.prevent="handleAuth" class="auth-form">
          <div class="form-group">
            <label><i class="fa-regular fa-user"></i> 用户名</label>
            <input v-model="authUsername" type="text" placeholder="请输入用户名" autocomplete="username" />
          </div>
          <div class="form-group">
            <label><i class="fa-solid fa-lock"></i> 密码</label>
            <div class="input-wrapper">
              <input v-model="authPassword" :type="showPassword ? 'text' : 'password'" placeholder="请输入密码" autocomplete="current-password" />
              <button type="button" class="toggle-pwd" @click="showPassword = !showPassword">
                <i :class="showPassword ? 'fa-regular fa-eye-slash' : 'fa-regular fa-eye'"></i>
              </button>
            </div>
          </div>
          <div v-if="loginTab === 'register'" class="form-group">
            <label><i class="fa-solid fa-lock"></i> 确认密码</label>
            <div class="input-wrapper">
              <input v-model="authConfirmPassword" :type="showConfirmPassword ? 'text' : 'password'" placeholder="请再次输入密码" autocomplete="new-password" />
              <button type="button" class="toggle-pwd" @click="showConfirmPassword = !showConfirmPassword">
                <i :class="showConfirmPassword ? 'fa-regular fa-eye-slash' : 'fa-regular fa-eye'"></i>
              </button>
            </div>
          </div>
          <button type="submit" class="submit-btn" :disabled="authLoading">
            <span v-if="authLoading" class="spinner"></span>
            <span v-else>{{ loginTab === 'login' ? '登 录' : '注 册' }}</span>
          </button>
        </form>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { useAuthStore } from '../store/auth'

const router = useRouter()
const authStore = useAuthStore()

const loginTab = ref('login')
const authUsername = ref('')
const authPassword = ref('')
const authConfirmPassword = ref('')
const authError = ref('')
const authLoading = ref(false)
const showPassword = ref(false)
const showConfirmPassword = ref(false)

const switchTab = (tab) => {
  loginTab.value = tab
  authError.value = ''
  authConfirmPassword.value = ''
}

const handleAuth = async () => {
  authError.value = ''
  if (!authUsername.value.trim() || !authPassword.value.trim()) {
    authError.value = '用户名和密码不能为空'
    return
  }
  if (loginTab.value === 'register') {
    if (authPassword.value.length < 4) {
      authError.value = '密码长度至少4个字符'
      return
    }
    if (authPassword.value !== authConfirmPassword.value) {
      authError.value = '两次输入的密码不一致'
      return
    }
  }
  authLoading.value = true
  try {
    if (loginTab.value === 'login') {
      await authStore.login(authUsername.value, authPassword.value)
    } else {
      await authStore.register(authUsername.value, authPassword.value)
    }
    authUsername.value = ''
    authPassword.value = ''
    authConfirmPassword.value = ''
    const redirect = localStorage.getItem('smartops_redirect')
    if (redirect) {
      localStorage.removeItem('smartops_redirect')
      router.push(redirect)
    } else {
      router.push('/chat')
    }
  } catch (e) {
    const msg = e.response?.data?.detail || e.message || '操作失败，请稍后重试'
    authError.value = msg
    // 错误持续显示，直到用户修改输入或切换标签
  } finally {
    authLoading.value = false
  }
}
</script>

<style scoped>
/* ===== 全屏覆盖式登录页 ===== */
.login-page {
  position: relative; width: 100vw; height: 100vh; overflow: hidden;
  background: #0a0f1e;
}

/* 动态背景 */
.login-bg {
  position: absolute; inset: 0; z-index: 0; overflow: hidden;
}
.bg-grid {
  position: absolute; inset: 0;
  background-image:
    linear-gradient(rgba(59,130,246,0.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(59,130,246,0.03) 1px, transparent 1px);
  background-size: 60px 60px;
}
.bg-glow {
  position: absolute; border-radius: 50%; filter: blur(120px); opacity: 0.5;
  animation: glowFloat 12s ease-in-out infinite;
}
.bg-glow-1 {
  width: 600px; height: 600px; top: -10%; left: -5%;
  background: radial-gradient(circle, rgba(59,130,246,0.2), transparent 70%);
}
.bg-glow-2 {
  width: 500px; height: 500px; bottom: -10%; right: -5%;
  background: radial-gradient(circle, rgba(139,92,246,0.15), transparent 70%);
  animation-delay: -6s;
}
@keyframes glowFloat {
  0%, 100% { transform: translate(0, 0) scale(1); }
  50% { transform: translate(30px, -20px) scale(1.1); }
}

/* 品牌信息层 - 全屏覆盖 */
.brand-layer {
  position: absolute; inset: 0; z-index: 1;
  display: flex; align-items: center; padding: 60px 80px;
}
.brand-content { max-width: 560px; }
.brand-logo { display: flex; align-items: center; gap: 14px; margin-bottom: 24px; }
.brand-logo-icon {
  background: linear-gradient(135deg, #3B82F6, #8B5CF6); width: 56px; height: 56px;
  border-radius: 16px; display: flex; align-items: center; justify-content: center;
  font-size: 28px; color: #fff; box-shadow: 0 4px 24px rgba(59,130,246,0.35);
}
.brand-logo-text { font-size: 42px; font-weight: 800; color: #f1f5f9; letter-spacing: -1px; }
.brand-tagline { font-size: 22px; color: #93c5fd; margin-bottom: 18px; font-weight: 500; }
.brand-desc { color: #94a3b8; line-height: 1.9; margin-bottom: 48px; font-size: 15px; max-width: 440px; }

.feature-item { display: flex; align-items: flex-start; gap: 14px; margin-bottom: 24px; }
.feature-icon {
  width: 46px; height: 46px; border-radius: 12px;
  background: rgba(59,130,246,0.1); border: 1px solid rgba(59,130,246,0.15);
  display: flex; align-items: center; justify-content: center; color: #60a5fa; flex-shrink: 0;
  backdrop-filter: blur(8px);
}
.feature-text { color: #cbd5e1; font-size: 14px; line-height: 1.7; padding-top: 2px; }
.feature-text strong { color: #e2e8f0; display: block; margin-bottom: 2px; font-size: 15px; }

/* 登录卡片浮层 - 居中偏右 */
.login-overlay {
  position: absolute; inset: 0; z-index: 2;
  display: flex; align-items: center; justify-content: flex-end;
  padding: 40px 80px;
  background: linear-gradient(90deg, transparent 0%, transparent 45%, rgba(10,15,30,0.6) 55%, rgba(10,15,30,0.85) 100%);
}

.login-card {
  width: 400px; padding: 36px 32px; border-radius: 20px;
  background: rgba(30,41,59,0.85); backdrop-filter: blur(24px);
  border: 1px solid rgba(255,255,255,0.08);
  box-shadow: 0 8px 40px rgba(0,0,0,0.4), 0 0 80px rgba(59,130,246,0.06);
}

.card-header { margin-bottom: 20px; }
.card-logo { display: flex; align-items: center; gap: 10px; }
.card-logo-icon {
  background: linear-gradient(135deg, #3B82F6, #8B5CF6); width: 32px; height: 32px;
  border-radius: 8px; display: flex; align-items: center; justify-content: center;
  font-size: 15px; color: #fff;
}
.card-logo-text { font-size: 18px; font-weight: 700; color: #f1f5f9; }

.login-card h2 { color: #f1f5f9; font-size: 24px; font-weight: 700; margin-bottom: 4px; }
.login-subtitle { color: #94a3b8; margin-bottom: 22px; font-size: 14px; }

.login-tabs {
  display: flex; gap: 0; margin-bottom: 22px; border-radius: 8px;
  overflow: hidden; border: 1px solid rgba(255,255,255,0.08);
  background: rgba(15,23,42,0.5);
}
.tab-btn {
  flex: 1; padding: 10px; border: none; background: transparent; color: #94a3b8;
  font-size: 14px; font-weight: 500; cursor: pointer; transition: all 0.25s;
}
.tab-btn.active { background: rgba(59,130,246,0.15); color: #60a5fa; }
.tab-btn:hover:not(.active) { background: rgba(255,255,255,0.04); color: #cbd5e1; }

.auth-error {
  background: rgba(239,68,68,0.1); border: 1px solid rgba(239,68,68,0.25);
  color: #fca5a5; padding: 10px 14px; border-radius: 8px; margin-bottom: 16px;
  font-size: 13px; display: flex; align-items: center; justify-content: space-between; gap: 8px;
}
.error-content { display: flex; align-items: center; gap: 8px; }
.error-close {
  background: none; border: none; color: #fca5a5; cursor: pointer;
  padding: 2px 6px; border-radius: 4px; font-size: 12px; opacity: 0.6; transition: opacity 0.15s;
}
.error-close:hover { opacity: 1; background: rgba(239,68,68,0.15); }

.error-fade-enter-active { animation: errorIn 0.3s ease; }
.error-fade-leave-active { animation: errorIn 0.2s ease reverse; }
@keyframes errorIn {
  from { opacity: 0; transform: translateY(-8px); }
  to { opacity: 1; transform: translateY(0); }
}

.auth-form { display: flex; flex-direction: column; gap: 16px; }
.form-group { display: flex; flex-direction: column; gap: 6px; }
.form-group label { font-size: 13px; color: #cbd5e1; font-weight: 500; display: flex; align-items: center; gap: 6px; }
.form-group label i { font-size: 12px; color: #64748b; }
.input-wrapper { position: relative; }
.form-group input {
  width: 100%; background: rgba(15,23,42,0.7); border: 1px solid rgba(255,255,255,0.1);
  border-radius: 10px; padding: 12px 14px; color: #f1f5f9; font-size: 14px;
  outline: none; transition: all 0.25s;
}
.input-wrapper input { padding-right: 42px; }
.form-group input::placeholder { color: #475569; }
.form-group input:focus {
  border-color: #3B82F6;
  box-shadow: 0 0 0 3px rgba(59,130,246,0.15);
  background: rgba(15,23,42,0.9);
}
.toggle-pwd {
  position: absolute; right: 4px; top: 50%; transform: translateY(-50%);
  background: none; border: none; color: #64748b; cursor: pointer;
  padding: 6px 8px; border-radius: 6px; transition: all 0.15s; font-size: 14px;
}
.toggle-pwd:hover { color: #94a3b8; background: rgba(255,255,255,0.05); }

.submit-btn {
  background: linear-gradient(135deg, #3B82F6, #6366F1); color: #fff; border: none;
  border-radius: 10px; padding: 13px; font-size: 15px; font-weight: 600;
  cursor: pointer; transition: all 0.25s; margin-top: 6px;
  display: flex; align-items: center; justify-content: center; gap: 8px;
  box-shadow: 0 4px 16px rgba(59,130,246,0.25);
}
.submit-btn:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 6px 24px rgba(59,130,246,0.4);
}
.submit-btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }

.spinner {
  width: 16px; height: 16px; border: 2px solid rgba(255,255,255,0.3);
  border-top-color: #fff; border-radius: 50%; animation: spin 0.6s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }

/* 响应式 */
@media (max-width: 900px) {
  .brand-layer { padding: 40px; }
  .login-overlay { padding: 20px; justify-content: center; background: rgba(10,15,30,0.8); }
  .login-card { width: 100%; max-width: 380px; }
  .brand-features { display: none; }
}
@media (max-width: 600px) {
  .brand-layer { display: none; }
  .login-overlay { background: #0a0f1e; }
}
</style>