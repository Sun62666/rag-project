<template>
  <div v-if="!authStore.isLoggedIn">
    <router-view />
  </div>
  <div v-else class="app-layout">
    <!-- 移动端遮罩 -->
    <div v-if="mobileMenuOpen" class="sidebar-overlay" @click="mobileMenuOpen = false"></div>
    <!-- 侧边栏 -->
    <div :class="['sidebar', { collapsed: sidebarCollapsed, 'mobile-open': mobileMenuOpen }]">
      <div class="sidebar-header">
        <div class="sidebar-logo">
          <span class="logo-icon"><i class="fa-solid fa-terminal"></i></span>
          <span class="logo-text">SmartOps</span>
          <button class="sidebar-toggle" @click="sidebarCollapsed = !sidebarCollapsed" :title="sidebarCollapsed ? '展开' : '收起'">
            <i :class="sidebarCollapsed ? 'fa-solid fa-angles-right' : 'fa-solid fa-angles-left'"></i>
          </button>
        </div>
        <button class="new-session-btn" @click="newSession">
          <i class="fa-solid fa-plus"></i> 新建会话
        </button>
      </div>
      <div class="sessions-list">
        <div
          v-for="sess in sessions"
          :key="sess.session_id"
          :class="['session-item', { active: sess.session_id === currentSessionId }]"
          @click="switchSession(sess.session_id)"
        >
          <i class="fa-regular fa-comment session-icon"></i>
          <div class="session-info">
            <template v-if="renamingSessionId === sess.session_id">
              <input
                v-model="renameValue"
                class="rename-input"
                @keyup.enter="confirmRename(sess.session_id)"
                @blur="confirmRename(sess.session_id)"
                @click.stop
              />
            </template>
            <template v-else>
              <div class="session-title" @dblclick="startRename(sess)">{{ sess.title }}</div>
              <div class="session-time">{{ sess.created_at || '' }}</div>
            </template>
          </div>
          <button class="session-del" @click.stop="deleteSession(sess.session_id)" title="删除">
            <i class="fa-solid fa-xmark"></i>
          </button>
        </div>
        <div v-if="sessions.length === 0" class="sessions-empty">
          <i class="fa-regular fa-comment-dots"></i>
          <span>暂无历史会话</span>
        </div>
      </div>
      <div class="sidebar-footer">
        <div class="user-info">
          <div class="user-avatar">{{ authStore.username?.[0]?.toUpperCase() }}</div>
          <div class="user-detail">
            <div class="user-name">{{ authStore.username }}</div>
            <div class="user-role">运维工程师</div>
          </div>
          <button class="logout-btn" @click="handleLogout" title="退出登录">
            <i class="fa-solid fa-right-from-bracket"></i>
          </button>
        </div>
      </div>
    </div>

    <!-- 主内容 -->
    <div class="main-area">
      <div class="top-bar">
        <div class="top-bar-left">
          <button class="mobile-menu-btn" @click="mobileMenuOpen = !mobileMenuOpen">
            <i class="fa-solid fa-bars"></i>
          </button>
          <span class="current-title">{{ currentTitle }}</span>
          <div class="view-nav">
            <router-link to="/dashboard" custom v-slot="{ isActive, navigate }">
              <button :class="['nav-btn', { active: isActive }]" @click="navigate">
                <i class="fa-solid fa-gauge-high"></i> 仪表盘
              </button>
            </router-link>
            <router-link to="/chat" custom v-slot="{ isActive, navigate }">
              <button :class="['nav-btn', { active: isActive }]" @click="navigate">
                <i class="fa-regular fa-comment-dots"></i> 智能对话
              </button>
            </router-link>
            <router-link to="/tools" custom v-slot="{ isActive, navigate }">
              <button :class="['nav-btn', { active: isActive }]" @click="navigate">
                <i class="fa-solid fa-screwdriver-wrench"></i> 运维工具
              </button>
            </router-link>
            <router-link to="/evaluate" custom v-slot="{ isActive, navigate }">
              <button :class="['nav-btn', { active: isActive }]" @click="navigate">
                <i class="fa-solid fa-chart-line"></i> 评估看板
              </button>
            </router-link>
          </div>
        </div>
        <div class="top-bar-right">
          <span :class="['mode-tag', currentMode === 'agent' ? 'agent' : 'graph']">
            {{ currentMode === 'agent' ? 'Agent' : 'Graph' }} 模式
          </span>
          <button class="mode-switch-btn" @click="toggleMode">切换模式</button>
        </div>
      </div>
      <div class="content-area">
        <router-view v-slot="{ Component }">
          <Transition name="page-fade" mode="out-in">
            <KeepAlive include="Chat,Evaluate">
              <component :is="Component" :key="$route.path" />
            </KeepAlive>
          </Transition>
        </router-view>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, watch, provide } from 'vue'
import { useRouter } from 'vue-router'
import { useAuthStore } from './store/auth'
import { sessionAPI, modeAPI } from './api'

const router = useRouter()
const authStore = useAuthStore()
let initialized = false

const sessions = ref([])
const currentSessionId = ref('')
const renamingSessionId = ref('')
const renameValue = ref('')
const currentMode = ref('agent')
const sidebarCollapsed = ref(false)
const mobileMenuOpen = ref(false)
const messageCache = ref({})

const currentTitle = computed(() => {
  const sess = sessions.value.find(s => s.session_id === currentSessionId.value)
  return sess ? sess.title : '新会话'
})

const initApp = async () => {
  if (initialized) return
  initialized = true
  await loadSessions()
  if (sessions.value.length > 0) currentSessionId.value = sessions.value[0].session_id
  else await newSession()
  await loadMode()
}

const loadSessions = async () => {
  try {
    const res = await sessionAPI.list()
    if (res.data.status === 'ok') sessions.value = res.data.sessions
  } catch (e) { console.error('加载会话列表失败:', e) }
}

const switchSession = async (sessId) => {
  currentSessionId.value = sessId
  mobileMenuOpen.value = false
}

const newSession = async () => {
  try {
    const res = await sessionAPI.create()
    currentSessionId.value = res.data.session_id
    await loadSessions()
  } catch (e) { currentSessionId.value = 'default' }
}

const deleteSession = async (sessId) => {
  try {
    await sessionAPI.delete(sessId)
    await loadSessions()
    if (sessId === currentSessionId.value) {
      if (sessions.value.length > 0) currentSessionId.value = sessions.value[0].session_id
      else await newSession()
    }
  } catch (e) { console.error('删除会话失败:', e) }
}

const startRename = (sess) => {
  renamingSessionId.value = sess.session_id
  renameValue.value = sess.title
}

const confirmRename = async (sessionId) => {
  const newTitle = renameValue.value.trim()
  renamingSessionId.value = ''
  if (!newTitle) return
  try {
    await sessionAPI.rename(sessionId, newTitle)
    await loadSessions()
  } catch (e) { console.error('重命名失败:', e) }
}

const handleLogout = async () => {
  // 清除所有前端状态
  sessions.value = []
  currentSessionId.value = ''
  messageCache.value = {}
  initialized = false
  await authStore.logout()
  router.push('/login')
}

const loadMode = async () => {
  try {
    const res = await modeAPI.get()
    currentMode.value = res.data.mode
  } catch (e) { console.error('获取模式失败:', e) }
}

const toggleMode = async () => {
  const newMode = currentMode.value === 'agent' ? false : true
  try {
    const res = await modeAPI.switch(newMode)
    currentMode.value = res.data.mode
  } catch (e) { console.error('切换模式失败:', e) }
}

watch(() => authStore.isLoggedIn, async (val) => {
  if (val && !initialized) await initApp()
})

onMounted(async () => {
  await authStore.checkAuth()
  if (authStore.isLoggedIn) await initApp()
})

provide('currentSessionId', currentSessionId)
provide('sessions', sessions)
provide('loadSessions', loadSessions)
provide('messageCache', messageCache)
</script>

<style>
/* ===== 全局暗色主题 ===== */
:root {
  --bg-primary: #0f172a;
  --bg-secondary: #1e293b;
  --bg-tertiary: #334155;
  --bg-card: #1e293b;
  --border: rgba(255,255,255,0.08);
  --border-light: rgba(255,255,255,0.04);
  --text-primary: #f1f5f9;
  --text-secondary: #cbd5e1;
  --text-muted: #94a3b8;
  --text-dim: #64748b;
  --accent: #3B82F6;
  --accent-hover: #2563EB;
  --accent-glow: rgba(59,130,246,0.25);
  --success: #22C55E;
  --warning: #EAB308;
  --danger: #EF4444;
}

* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'PingFang SC', 'Microsoft YaHei', sans-serif;
  background: var(--bg-primary); color: var(--text-primary);
}

/* 滚动条 */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--bg-tertiary); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #475569; }

/* Element Plus 暗色覆盖 */
.el-card { background: var(--bg-card) !important; border-color: var(--border) !important; color: var(--text-primary) !important; }
.el-card__header { border-bottom-color: var(--border) !important; color: var(--text-primary) !important; }
.el-table { --el-table-bg-color: var(--bg-secondary) !important; --el-table-tr-bg-color: var(--bg-secondary) !important; --el-table-header-bg-color: var(--bg-tertiary) !important; --el-table-row-hover-bg-color: rgba(59,130,246,0.08) !important; --el-table-text-color: var(--text-secondary) !important; --el-table-header-text-color: var(--text-primary) !important; --el-table-border-color: var(--border) !important; }
.el-tabs--border-card { background: var(--bg-secondary) !important; border-color: var(--border) !important; }
.el-tabs--border-card > .el-tabs__header { background: var(--bg-tertiary) !important; border-bottom-color: var(--border) !important; }
.el-tabs--border-card > .el-tabs__header .el-tabs__item { color: var(--text-muted) !important; }
.el-tabs--border-card > .el-tabs__header .el-tabs__item.is-active { color: var(--accent) !important; background: var(--bg-secondary) !important; }
.el-input__wrapper { background: var(--bg-primary) !important; box-shadow: 0 0 0 1px var(--border) inset !important; }
.el-input__inner { color: var(--text-primary) !important; }
.el-input__inner::placeholder { color: var(--text-dim) !important; }
.el-select .el-input__wrapper { background: var(--bg-primary) !important; }
.el-textarea__inner { background: var(--bg-primary) !important; color: var(--text-primary) !important; border-color: var(--border) !important; }
.el-button--default { background: var(--bg-tertiary) !important; border-color: var(--border) !important; color: var(--text-secondary) !important; }
.el-button--default:hover { border-color: var(--accent) !important; color: var(--accent) !important; }
.el-tag { border-color: var(--border) !important; }
.el-progress__text { color: var(--text-secondary) !important; }
.el-upload-dragger { background: var(--bg-primary) !important; border-color: var(--border) !important; color: var(--text-secondary) !important; }
.el-upload-dragger:hover { border-color: var(--accent) !important; }
.el-divider { border-color: var(--border) !important; }
.el-alert { border-color: var(--border) !important; }
.el-form-item__label { color: var(--text-secondary) !important; }
.el-input-number { --el-input-number-unit-offset-x: 0 !important; }

/* ===== 布局 ===== */
.app-layout { display: flex; height: 100vh; background: var(--bg-primary); }

.sidebar {
  width: 260px; background: var(--bg-secondary); border-right: 1px solid var(--border);
  display: flex; flex-direction: column; flex-shrink: 0;
}
.sidebar-header { padding: 16px; border-bottom: 1px solid var(--border); }
.sidebar-logo { display: flex; align-items: center; gap: 10px; margin-bottom: 14px; }
.logo-icon {
  background: linear-gradient(135deg, #3B82F6, #8B5CF6); color: #fff;
  width: 32px; height: 32px; border-radius: 8px;
  display: flex; align-items: center; justify-content: center; font-size: 15px;
}
.logo-text { font-size: 18px; font-weight: 700; color: var(--text-primary); }

.new-session-btn {
  width: 100%; padding: 9px; border-radius: 8px; border: 1px dashed var(--border);
  background: transparent; color: var(--text-secondary); font-size: 13px;
  cursor: pointer; transition: all 0.2s; display: flex; align-items: center; justify-content: center; gap: 6px;
}
.new-session-btn:hover { border-color: var(--accent); color: var(--accent); background: rgba(59,130,246,0.06); }

.sessions-list { flex: 1; overflow-y: auto; padding: 8px; }
.session-item {
  display: flex; align-items: center; padding: 10px 12px; border-radius: 8px;
  cursor: pointer; transition: all 0.15s; margin-bottom: 2px; gap: 8px;
}
.session-item:hover { background: rgba(255,255,255,0.04); }
.session-item.active { background: rgba(59,130,246,0.12); }
.session-icon { color: var(--text-dim); font-size: 14px; flex-shrink: 0; }
.session-item.active .session-icon { color: var(--accent); }
.session-info { flex: 1; min-width: 0; }
.session-title { font-size: 13px; color: var(--text-secondary); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.session-item.active .session-title { color: var(--text-primary); }
.session-time { font-size: 11px; color: var(--text-dim); margin-top: 2px; }
.rename-input {
  background: var(--bg-primary); border: 1px solid var(--accent); border-radius: 4px;
  padding: 2px 6px; color: var(--text-primary); font-size: 13px; width: 100%; outline: none;
}
.session-del {
  background: none; border: none; color: var(--text-dim); cursor: pointer;
  padding: 4px; border-radius: 4px; opacity: 0; transition: all 0.15s; font-size: 12px;
}
.session-item:hover .session-del { opacity: 1; }
.session-del:hover { color: var(--danger); background: rgba(239,68,68,0.1); }

.sessions-empty {
  text-align: center; color: var(--text-dim); padding: 40px 0; font-size: 13px;
  display: flex; flex-direction: column; align-items: center; gap: 8px;
}

.sidebar-footer { padding: 12px 16px; border-top: 1px solid var(--border); }
.user-info { display: flex; align-items: center; gap: 10px; }
.user-avatar {
  width: 34px; height: 34px; border-radius: 50%; background: linear-gradient(135deg, #3B82F6, #6366F1);
  display: flex; align-items: center; justify-content: center; color: #fff; font-weight: 600; font-size: 14px;
}
.user-detail { flex: 1; }
.user-name { font-size: 13px; font-weight: 500; color: var(--text-primary); }
.user-role { font-size: 11px; color: var(--text-dim); }
.logout-btn {
  background: none; border: none; color: var(--text-dim); cursor: pointer;
  padding: 6px; border-radius: 6px; transition: all 0.15s; font-size: 14px;
}
.logout-btn:hover { color: var(--danger); background: rgba(239,68,68,0.1); }

.main-area { flex: 1; display: flex; flex-direction: column; min-width: 0; background: var(--bg-primary); }
.top-bar {
  display: flex; align-items: center; justify-content: space-between;
  padding: 10px 20px; background: var(--bg-secondary); border-bottom: 1px solid var(--border);
}
.top-bar-left { display: flex; align-items: center; gap: 16px; }
.current-title { font-weight: 600; font-size: 15px; color: var(--text-primary); }
.view-nav { display: flex; gap: 4px; }
.nav-btn {
  padding: 7px 14px; border-radius: 6px; border: none; background: transparent;
  color: var(--text-muted); font-size: 13px; cursor: pointer; transition: all 0.15s;
  display: flex; align-items: center; gap: 6px;
}
.nav-btn:hover { background: rgba(255,255,255,0.04); color: var(--text-secondary); }
.nav-btn.active { background: rgba(59,130,246,0.12); color: var(--accent); }

.top-bar-right { display: flex; align-items: center; gap: 8px; }
.mode-tag {
  padding: 4px 10px; border-radius: 4px; font-size: 12px; font-weight: 600;
}
.mode-tag.agent { background: rgba(34,197,94,0.12); color: #4ade80; }
.mode-tag.graph { background: rgba(234,179,8,0.12); color: #facc15; }
.mode-switch-btn {
  padding: 6px 12px; border-radius: 6px; border: 1px solid var(--border);
  background: transparent; color: var(--text-secondary); font-size: 12px;
  cursor: pointer; transition: all 0.15s;
}
.mode-switch-btn:hover { border-color: var(--accent); color: var(--accent); }

.content-area { flex: 1; min-height: 0; overflow-y: auto; display: flex; flex-direction: column; }

/* 路由切换过渡 */
.page-fade-enter-active { animation: pageIn 0.25s ease; }
.page-fade-leave-active { animation: pageIn 0.15s ease reverse; }
@keyframes pageIn {
  from { opacity: 0; transform: translateY(8px); }
  to { opacity: 1; transform: translateY(0); }
}

/* 侧边栏折叠 */
.sidebar.collapsed { width: 64px; }
.sidebar.collapsed .logo-text,
.sidebar.collapsed .new-session-btn span,
.sidebar.collapsed .session-info,
.sidebar.collapsed .user-detail,
.sidebar.collapsed .session-del { display: none; }
.sidebar.collapsed .new-session-btn { padding: 9px 0; }
.sidebar.collapsed .sidebar-header { padding: 12px 8px; }
.sidebar.collapsed .session-item { padding: 10px 8px; justify-content: center; }
.sidebar.collapsed .sidebar-footer { padding: 12px 8px; }
.sidebar.collapsed .user-info { justify-content: center; }

.sidebar-toggle {
  background: none; border: none; color: var(--text-dim); cursor: pointer;
  padding: 6px; border-radius: 6px; transition: all 0.15s; font-size: 14px;
}
.sidebar-toggle:hover { color: var(--text-secondary); background: rgba(255,255,255,0.04); }

/* 移动端适配 */
@media (max-width: 768px) {
  .sidebar {
    position: fixed; left: 0; top: 0; bottom: 0; z-index: 100;
    transform: translateX(-100%); transition: transform 0.3s ease;
  }
  .sidebar.mobile-open { transform: translateX(0); }
  .sidebar-overlay {
    position: fixed; inset: 0; z-index: 99; background: rgba(0,0,0,0.5);
  }
  .view-nav { display: none; }
  .mobile-menu-btn { display: flex !important; }
}
.mobile-menu-btn {
  display: none; background: none; border: none; color: var(--text-secondary);
  cursor: pointer; padding: 6px 8px; border-radius: 6px; font-size: 18px;
}
</style>