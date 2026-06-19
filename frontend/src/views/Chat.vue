<template>
  <div class="chat-view">
    <div class="chat-container" ref="chatRef" @scroll="checkScrollPosition">
      <div v-for="(msg, idx) in messages" :key="idx" :class="['message-row', msg.role]">
        <!-- 助手消息：左侧 -->
        <template v-if="msg.role === 'assistant'">
          <div class="avatar assistant-avatar"><i class="fa-solid fa-robot"></i></div>
          <div class="message-body assistant-body">
            <div class="bubble assistant-bubble" :class="{ cached: msg.cached }">
              <div v-if="msg.content" v-html="sanitizeHtml(renderMarkdown(msg.content))" class="markdown-body"></div>
              <span v-if="msg.content === '' && isLoading && idx === messages.length - 1" class="typing-indicator">
                <span class="dot"></span><span class="dot"></span><span class="dot"></span>
              </span>
            </div>
            <div class="message-meta">
              <span class="msg-time">{{ formatTime(msg.time) }}</span>
              <button v-if="msg.content" class="action-btn" @click="copyMessage(msg.content, $event)" title="复制">
                <i class="fa-regular fa-copy"></i>
              </button>
            </div>
          </div>
        </template>
        <!-- 用户消息：右侧 -->
        <template v-else>
          <div class="message-body user-body">
            <div class="bubble user-bubble">{{ msg.content }}</div>
            <div class="message-meta user-meta">
              <span class="msg-time">{{ formatTime(msg.time) }}</span>
            </div>
          </div>
          <div class="avatar user-avatar"><i class="fa-solid fa-user"></i></div>
        </template>
      </div>
      <div v-if="messages.length === 0" class="empty-state">
        <div class="empty-icon"><i class="fa-solid fa-magnifying-glass-chart"></i></div>
        <div class="empty-title">智能运维助手</div>
        <div class="empty-desc">输入运维问题，或点击下方快捷问题开始</div>
        <div class="quick-questions">
          <button v-for="q in quickQuestions" :key="q.text" class="quick-btn" @click="quickAsk(q.text)">
            <i :class="q.icon"></i> {{ q.label }}
          </button>
        </div>
      </div>
    </div>

    <div class="status-bar" :class="statusClass" v-if="statusText">
      {{ statusText }}
    </div>

    <div class="input-area">
      <div class="input-wrapper" :class="{ expanded: inputExpanded }">
        <textarea
          v-model="query"
          placeholder="输入运维问题... (Enter 发送, Shift+Enter 换行)"
          :rows="inputExpanded ? 12 : 2"
          @keydown.enter.exact.prevent="sendQuery"
          :disabled="isLoading"
          class="chat-input"
          ref="chatInputRef"
        ></textarea>
        <div class="input-actions">
          <button v-if="isInputLong" class="expand-btn" @click="inputExpanded = !inputExpanded" :title="inputExpanded ? '收起' : '展开'">
            <i :class="inputExpanded ? 'fa-solid fa-chevron-down' : 'fa-solid fa-chevron-up'"></i>
          </button>
          <button class="send-btn" :disabled="isLoading || !query.trim()" @click="sendQuery">
            <i class="fa-solid fa-paper-plane"></i>
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, nextTick, inject, watch, onMounted, onBeforeUnmount, onActivated, onDeactivated } from 'vue'
import { useRoute } from 'vue-router'
import { chatAPI, sessionAPI } from '../api'
import { marked } from 'marked'
import DOMPurify from 'dompurify'

defineOptions({ name: 'Chat' })

marked.setOptions({ breaks: true, gfm: true })

const route = useRoute()
const currentSessionId = inject('currentSessionId')
const loadSessions = inject('loadSessions')

const query = ref('')
const messages = ref([])
const isLoading = ref(false)
const statusText = ref('')
const statusClass = ref('')
const chatRef = ref(null)
const chatInputRef = ref(null)
const inputExpanded = ref(false)
const messageCache = inject('messageCache', ref({}))
const userScrolledUp = ref(false)  // 用户是否手动上滑

const isInputLong = computed(() => {
  if (!query.value) return false
  const lines = query.value.split('\n').length
  return lines > 3 || query.value.length > 150
})

watch(isInputLong, (val) => {
  if (!val) inputExpanded.value = false
})

const quickQuestions = [
  { text: 'Linux服务器CPU使用率持续100%如何排查？', label: 'CPU使用率过高', icon: 'fa-solid fa-microchip' },
  { text: '如何查看系统日志定位问题？', label: '查看系统日志', icon: 'fa-solid fa-file-lines' },
  { text: '磁盘空间不足怎么处理？', label: '磁盘空间不足', icon: 'fa-solid fa-hard-drive' },
  { text: 'Docker容器无法启动怎么排查？', label: 'Docker容器问题', icon: 'fa-brands fa-docker' },
]

const formatTime = (time) => {
  if (!time) return ''
  const d = new Date(time)
  return d.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })
}

const scrollToBottom = async () => {
  await nextTick()
  if (chatRef.value) {
    chatRef.value.scrollTo({ top: chatRef.value.scrollHeight, behavior: 'smooth' })
  }
}

// 检测用户是否在底部附近（允许自动滚动），还是上滑查看历史（停止自动滚动）
const checkScrollPosition = () => {
  if (!chatRef.value) return
  const { scrollTop, scrollHeight, clientHeight } = chatRef.value
  // 距离底部 80px 以内视为"在底部"，允许自动滚动
  const isNearBottom = scrollHeight - scrollTop - clientHeight < 80
  userScrolledUp.value = !isNearBottom
}

// 智能滚动：仅在用户未上滑时自动滚动到底部
const smartScrollToBottom = async () => {
  if (!userScrolledUp.value) {
    await scrollToBottom()
  }
}

const renderMarkdown = (text) => {
  if (!text) return ''
  try { return marked.parse(text) } catch { return text }
}

const sanitizeHtml = (html) => {
  return DOMPurify.sanitize(html, {
    ALLOWED_TAGS: ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'br', 'hr',
      'ul', 'ol', 'li', 'blockquote', 'pre', 'code', 'em', 'strong',
      'a', 'table', 'thead', 'tbody', 'tr', 'th', 'td', 'img', 'span', 'div'],
    ALLOWED_ATTR: ['href', 'target', 'class', 'src', 'alt', 'title'],
  })
}

const copyMessage = async (content, event) => {
  const btn = event.currentTarget
  try {
    await navigator.clipboard.writeText(content)
    btn.innerHTML = '<i class="fa-solid fa-check"></i>'
    setTimeout(() => { btn.innerHTML = '<i class="fa-regular fa-copy"></i>' }, 1500)
  } catch {
    try {
      const ta = document.createElement('textarea')
      ta.value = content
      ta.style.position = 'fixed'
      ta.style.left = '-9999px'
      document.body.appendChild(ta)
      ta.select()
      document.execCommand('copy')
      document.body.removeChild(ta)
    } catch (e) { console.warn('复制失败:', e) }
  }
}

const quickAsk = (q) => {
  query.value = q
  sendQuery()
}

const loadHistory = async (sessionId) => {
  if (messageCache.value[sessionId]) {
    messages.value = [...messageCache.value[sessionId]]
    await scrollToBottom()
    return
  }
  try {
    const res = await sessionAPI.get(sessionId)
    if (res.data.status === 'ok' && res.data.history?.length > 0) {
      messages.value = res.data.history.map(h => ({ role: h.role, content: h.content, time: h.time || '' }))
    } else {
      messages.value = []
    }
  } catch { messages.value = [] }
  await scrollToBottom()
}

const sendQuery = async () => {
  if (!query.value.trim() || isLoading.value) return
  isLoading.value = true
  const userQuery = query.value
  query.value = ''
  inputExpanded.value = false
  const sid = currentSessionId.value

  messages.value.push({ role: 'user', content: userQuery, time: Date.now() })
  messages.value.push({ role: 'assistant', content: '', cached: false, time: Date.now() })
  statusText.value = '⏳ 正在连接...'
  statusClass.value = 'loading'
  userScrolledUp.value = false  // 新消息发送时重置，允许自动滚动
  await scrollToBottom()

  const assistantMsg = messages.value[messages.value.length - 1]

  try {
    const response = await chatAPI.ask(userQuery, sid)
    if (!response.ok) throw new Error(`HTTP ${response.status}`)

    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n\n')
      buffer = lines.pop() || ''

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            const data = JSON.parse(line.slice(6))
            if (data.type === 'status') {
              statusText.value = data.message
            } else if (data.type === 'token') {
              assistantMsg.content += data.content
              smartScrollToBottom()
            } else if (data.type === 'done') {
              assistantMsg.cached = data.from_cache || false
              statusText.value = data.from_cache ? '✅ 已从缓存中检索到解决方案' : '✅ 已完成检索调用输出解决方案'
              statusClass.value = 'done'
            }
          } catch (e) { console.warn('SSE JSON 解析失败:', e, line.slice(0, 80)) }
        }
      }
    }
    messageCache.value[sid] = [...messages.value]
    await loadSessions()
  } catch (error) {
    assistantMsg.content = `请求失败: ${error.message}`
    statusText.value = `❌ 错误: ${error.message}`
    statusClass.value = 'error'
  } finally {
    isLoading.value = false
  }
}

watch(currentSessionId, (newSid, oldSid) => {
  if (oldSid && messages.value.length > 0) {
    messageCache.value[oldSid] = [...messages.value]
  }
  if (newSid) loadHistory(newSid)
}, { immediate: true })

onMounted(() => {
  const q = route.query.q
  if (q) {
    query.value = decodeURIComponent(q)
    route.query = {}
    nextTick(() => sendQuery())
  }
})

onBeforeUnmount(() => {
  const sid = currentSessionId.value
  if (sid && messages.value.length > 0) {
    messageCache.value[sid] = [...messages.value]
  }
})

onActivated(() => {
  const sid = currentSessionId.value
  if (sid && messageCache.value[sid]) {
    messages.value = [...messageCache.value[sid]]
  }
  scrollToBottom()
})

onDeactivated(() => {
  const sid = currentSessionId.value
  if (sid && messages.value.length > 0) {
    messageCache.value[sid] = [...messages.value]
  }
})
</script>

<style scoped>
/* ===== 基础布局 ===== */
.chat-view {
  display: flex; flex-direction: column; height: 100%;
  background: var(--bg-primary);
}
.chat-container {
  flex: 1; overflow-y: auto; padding: 28px 0;
  scroll-behavior: smooth;
}

/* ===== 消息行 ===== */
.message-row {
  display: flex; gap: 14px; margin-bottom: 28px; align-items: flex-start;
  padding: 0 48px;
  animation: msgFadeIn 0.35s ease-out;
}
@keyframes msgFadeIn {
  from { opacity: 0; transform: translateY(8px); }
  to { opacity: 1; transform: translateY(0); }
}

.message-row.assistant { justify-content: flex-start; }
.message-row.user { justify-content: flex-end; }

/* ===== 头像 ===== */
.avatar {
  width: 40px; height: 40px; border-radius: 50%; display: flex;
  align-items: center; justify-content: center; font-size: 16px; flex-shrink: 0;
  box-shadow: 0 3px 10px rgba(0,0,0,0.25);
  margin-top: 2px;
}
.assistant-avatar {
  background: linear-gradient(135deg, #8B5CF6, #A855F7);
  color: #fff;
}
.user-avatar {
  background: linear-gradient(135deg, #667eea, #764ba2);
  color: #fff;
}

/* ===== 消息体 ===== */
.message-body { max-width: 75%; min-width: 60px; }

/* ===== 气泡 ===== */
.bubble {
  padding: 14px 20px; line-height: 1.8; font-size: 14px;
  word-break: break-word; letter-spacing: 0.2px;
  position: relative;
}

/* 助手气泡：半透明深色背景 + 左下锐角 */
.assistant-bubble {
  background: rgba(255,255,255,0.08);
  color: var(--text-primary);
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 18px 18px 4px 18px;
  box-shadow: 0 2px 12px rgba(0,0,0,0.15);
  backdrop-filter: blur(8px);
}
.assistant-bubble.cached { border-color: rgba(34,197,94,0.35); }

/* 用户气泡：蓝紫渐变 + 右下锐角 */
.user-bubble {
  background: linear-gradient(135deg, #667eea, #764ba2);
  color: #fff;
  border-radius: 18px 18px 18px 4px;
  box-shadow: 0 4px 18px rgba(102,126,234,0.3);
}

/* ===== 消息元信息（时间戳+操作） ===== */
.message-meta {
  display: flex; align-items: center; gap: 8px;
  margin-top: 6px; padding-left: 4px;
}
.user-meta {
  justify-content: flex-end; padding-right: 4px; padding-left: 0;
}
.msg-time {
  font-size: 11px; color: var(--text-dim); letter-spacing: 0.3px;
}
.action-btn {
  background: none; border: none; color: var(--text-dim); cursor: pointer;
  padding: 3px 7px; border-radius: 6px; font-size: 12px; transition: all 0.2s;
}
.action-btn:hover { color: var(--accent); background: rgba(59,130,246,0.12); }

/* ===== 打字动画 ===== */
.typing-indicator { display: inline-flex; gap: 6px; padding: 6px 0; }
.dot {
  width: 8px; height: 8px; border-radius: 50%; background: var(--accent);
  animation: blink 1.4s infinite;
}
.dot:nth-child(2) { animation-delay: 0.2s; }
.dot:nth-child(3) { animation-delay: 0.4s; }
@keyframes blink { 0%, 80%, 100% { opacity: 0.3; } 40% { opacity: 1; } }

/* ===== 空状态 ===== */
.empty-state { text-align: center; padding: 80px 20px; }
.empty-icon { font-size: 56px; color: var(--accent); margin-bottom: 20px; opacity: 0.8; }
.empty-title { font-size: 26px; font-weight: 700; color: var(--text-primary); margin-bottom: 8px; }
.empty-desc { color: var(--text-muted); margin-bottom: 32px; font-size: 15px; }
.quick-questions { display: flex; flex-wrap: wrap; gap: 10px; justify-content: center; }
.quick-btn {
  padding: 10px 18px; border-radius: 20px; border: 1px solid var(--border);
  background: var(--bg-secondary); color: var(--text-secondary); font-size: 13px;
  cursor: pointer; transition: all 0.2s; display: flex; align-items: center; gap: 8px;
}
.quick-btn:hover {
  border-color: var(--accent); color: var(--accent);
  background: rgba(59,130,246,0.08); transform: translateY(-1px);
}

/* ===== 状态栏 ===== */
.status-bar {
  padding: 8px 24px; font-size: 13px; background: var(--bg-secondary);
  border-top: 1px solid var(--border);
}
.status-bar.loading { color: var(--accent); }
.status-bar.done { color: var(--success); }
.status-bar.error { color: var(--danger); }

/* ===== 输入区 ===== */
.input-area {
  padding: 14px 24px 18px; background: var(--bg-secondary);
  border-top: 1px solid var(--border);
  flex-shrink: 0;
}
.input-wrapper {
  width: 100%; display: flex; gap: 10px; align-items: flex-end;
  background: var(--bg-primary); border: 1px solid var(--border);
  border-radius: 12px; padding: 8px 8px 8px 16px;
  transition: border-color 0.2s;
}
.input-wrapper.expanded { align-items: flex-start; }
.input-wrapper:focus-within { border-color: var(--accent); }
.chat-input {
  flex: 1; min-width: 0; background: transparent; border: none; outline: none;
  color: var(--text-primary); font-size: 14px; font-family: inherit;
  line-height: 1.5; resize: none;
}
.chat-input::placeholder { color: var(--text-dim); }

.input-actions {
  display: flex; flex-direction: column; gap: 6px; flex-shrink: 0;
}
.expand-btn {
  width: 32px; height: 32px; border-radius: 8px; border: 1px solid var(--border);
  background: var(--bg-secondary); color: var(--text-dim); cursor: pointer;
  display: flex; align-items: center; justify-content: center;
  font-size: 12px; transition: all 0.2s;
}
.expand-btn:hover { color: var(--accent); border-color: var(--accent); background: rgba(59,130,246,0.08); }

.send-btn {
  width: 40px; height: 40px; border-radius: 50%; border: none; flex-shrink: 0;
  background: linear-gradient(135deg, #667eea, #764ba2); color: #fff;
  cursor: pointer; display: flex; align-items: center; justify-content: center;
  font-size: 15px; transition: opacity 0.15s;
}
.send-btn:hover:not(:disabled) { opacity: 0.85; }
.send-btn:disabled { opacity: 0.3; cursor: not-allowed; }

/* ===== Markdown 渲染样式 ===== */
.markdown-body :deep(h1), .markdown-body :deep(h2), .markdown-body :deep(h3) {
  color: var(--text-primary); margin: 16px 0 8px; font-weight: 600;
}
.markdown-body :deep(h1) { font-size: 20px; }
.markdown-body :deep(h2) { font-size: 17px; }
.markdown-body :deep(h3) { font-size: 15px; }
.markdown-body :deep(p) { margin-bottom: 10px; color: var(--text-secondary); }
.markdown-body :deep(ul), .markdown-body :deep(ol) { padding-left: 20px; margin-bottom: 10px; color: var(--text-secondary); }
.markdown-body :deep(li) { margin-bottom: 4px; }
.markdown-body :deep(blockquote) {
  border-left: 3px solid var(--accent); padding: 8px 16px; margin: 10px 0;
  background: rgba(59,130,246,0.06); border-radius: 0 6px 6px 0; color: var(--text-secondary);
}
.markdown-body :deep(pre) {
  background: #0f172a; border: 1px solid var(--border); border-radius: 8px;
  padding: 16px; overflow-x: auto; margin: 12px 0;
}
.markdown-body :deep(code) {
  font-family: 'Fira Code', 'Cascadia Code', 'Consolas', monospace; font-size: 13px;
}
.markdown-body :deep(:not(pre) > code) {
  background: rgba(59,130,246,0.1); padding: 2px 6px; border-radius: 4px;
  color: #60a5fa; font-size: 13px;
}
.markdown-body :deep(pre code) { color: #e2e8f0; }
.markdown-body :deep(a) { color: var(--accent); text-decoration: none; }
.markdown-body :deep(a:hover) { text-decoration: underline; }
.markdown-body :deep(table) { border-collapse: collapse; width: 100%; margin: 12px 0; }
.markdown-body :deep(th), .markdown-body :deep(td) {
  border: 1px solid var(--border); padding: 8px 12px; text-align: left; color: var(--text-secondary);
}
.markdown-body :deep(th) { background: var(--bg-tertiary); color: var(--text-primary); font-weight: 600; }
.markdown-body :deep(strong) { color: var(--text-primary); }
.markdown-body :deep(em) { color: var(--text-secondary); }

/* ===== 响应式：移动端 ===== */
@media (max-width: 768px) {
  .message-row { padding: 0 16px; gap: 10px; }
  .message-body { max-width: 88%; }
  .avatar { width: 32px; height: 32px; font-size: 14px; }
  .bubble { padding: 12px 14px; font-size: 13px; }
  .input-area { padding: 12px 12px 16px; }
}
</style>
