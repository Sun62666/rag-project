<template>
  <div class="chat-view">
    <div class="chat-container" ref="chatRef">
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
            <div v-if="msg.content" class="message-actions">
              <button class="action-btn" @click="copyMessage(msg.content, $event)" title="复制">
                <i class="fa-regular fa-copy"></i>
              </button>
            </div>
          </div>
        </template>
        <!-- 用户消息：右侧 -->
        <template v-else>
          <div class="message-body user-body">
            <div class="bubble user-bubble">{{ msg.content }}</div>
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
      <textarea
        v-model="query"
        placeholder="输入运维问题... (Enter 发送, Shift+Enter 换行)"
        rows="2"
        @keydown.enter.exact.prevent="sendQuery"
        :disabled="isLoading"
        class="chat-input"
      ></textarea>
      <button class="send-btn" :disabled="isLoading || !query.trim()" @click="sendQuery">
        <i class="fa-solid fa-paper-plane"></i>
      </button>
    </div>
  </div>
</template>

<script setup>
import { ref, nextTick, inject, watch, onMounted, onBeforeUnmount, onActivated } from 'vue'
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
const messageCache = inject('messageCache', ref({}))

const quickQuestions = [
  { text: 'Linux服务器CPU使用率持续100%如何排查？', label: 'CPU使用率过高', icon: 'fa-solid fa-microchip' },
  { text: '如何查看系统日志定位问题？', label: '查看系统日志', icon: 'fa-solid fa-file-lines' },
  { text: '磁盘空间不足怎么处理？', label: '磁盘空间不足', icon: 'fa-solid fa-hard-drive' },
  { text: 'Docker容器无法启动怎么排查？', label: 'Docker容器问题', icon: 'fa-brands fa-docker' },
]

const scrollToBottom = async () => {
  await nextTick()
  if (chatRef.value) chatRef.value.scrollTop = chatRef.value.scrollHeight
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
      messages.value = res.data.history.map(h => ({ role: h.role, content: h.content }))
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
  const sid = currentSessionId.value

  messages.value.push({ role: 'user', content: userQuery })
  messages.value.push({ role: 'assistant', content: '', cached: false })
  statusText.value = '⏳ 正在连接...'
  statusClass.value = 'loading'
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
              scrollToBottom()
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
  // 切换前保存当前会话的消息到缓存
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

// 组件卸载时保存当前会话消息到缓存
onBeforeUnmount(() => {
  const sid = currentSessionId.value
  if (sid && messages.value.length > 0) {
    messageCache.value[sid] = [...messages.value]
  }
})

// 从 KeepAlive 缓存恢复时滚动到底部
onActivated(() => {
  scrollToBottom()
})
</script>

<style scoped>
.chat-view { display: flex; flex-direction: column; height: 100%; background: var(--bg-primary); }
.chat-container { flex: 1; overflow-y: auto; padding: 24px 32px; }

/* ===== 消息布局: 助手左, 用户右 ===== */
.message-row { display: flex; gap: 12px; margin-bottom: 24px; align-items: flex-start; }

/* 助手消息：靠左 */
.message-row.assistant { justify-content: flex-start; }
/* 用户消息：靠右 */
.message-row.user { justify-content: flex-end; }

.avatar {
  width: 38px; height: 38px; border-radius: 50%; display: flex;
  align-items: center; justify-content: center; font-size: 16px; flex-shrink: 0;
}
.assistant-avatar { background: linear-gradient(135deg, #8B5CF6, #A855F7); color: #fff; }
.user-avatar { background: linear-gradient(135deg, #3B82F6, #6366F1); color: #fff; }

.message-body { max-width: 72%; min-width: 0; }

.bubble {
  padding: 14px 18px; border-radius: 14px; line-height: 1.75; font-size: 14px;
  word-break: break-word;
}
.assistant-bubble {
  background: var(--bg-secondary); color: var(--text-primary);
  border: 1px solid var(--border); border-bottom-left-radius: 4px;
}
.user-bubble {
  background: linear-gradient(135deg, #3B82F6, #6366F1); color: #fff;
  border-bottom-right-radius: 4px;
}
.bubble.cached { border-color: rgba(34,197,94,0.4); }

.message-actions { margin-top: 6px; display: flex; gap: 4px; }
.action-btn {
  background: none; border: none; color: var(--text-dim); cursor: pointer;
  padding: 4px 8px; border-radius: 4px; font-size: 13px; transition: all 0.15s;
}
.action-btn:hover { color: var(--accent); background: rgba(59,130,246,0.1); }

/* 打字动画 */
.typing-indicator { display: inline-flex; gap: 5px; padding: 4px 0; }
.dot {
  width: 8px; height: 8px; border-radius: 50%; background: var(--accent);
  animation: blink 1.4s infinite;
}
.dot:nth-child(2) { animation-delay: 0.2s; }
.dot:nth-child(3) { animation-delay: 0.4s; }
@keyframes blink { 0%, 80%, 100% { opacity: 0.3; } 40% { opacity: 1; } }

/* 空状态 */
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
.quick-btn:hover { border-color: var(--accent); color: var(--accent); background: rgba(59,130,246,0.08); transform: translateY(-1px); }

/* 状态栏 */
.status-bar {
  padding: 8px 24px; font-size: 13px; background: var(--bg-secondary);
  border-top: 1px solid var(--border);
}
.status-bar.loading { color: var(--accent); }
.status-bar.done { color: var(--success); }
.status-bar.error { color: var(--danger); }

/* 输入区 */
.input-area {
  display: flex; gap: 12px; padding: 16px 24px; background: var(--bg-secondary);
  border-top: 1px solid var(--border); align-items: flex-end;
}
.chat-input {
  flex: 1; background: var(--bg-primary); border: 1px solid var(--border);
  border-radius: 10px; padding: 12px 16px; color: var(--text-primary);
  font-size: 14px; resize: none; outline: none; line-height: 1.5;
  font-family: inherit;
}
.chat-input::placeholder { color: var(--text-dim); }
.chat-input:focus { border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-glow); }

.send-btn {
  width: 44px; height: 44px; border-radius: 10px; border: none;
  background: linear-gradient(135deg, #3B82F6, #6366F1); color: #fff;
  cursor: pointer; transition: all 0.2s; display: flex; align-items: center;
  justify-content: center; font-size: 16px; flex-shrink: 0;
}
.send-btn:hover:not(:disabled) { transform: translateY(-1px); box-shadow: 0 4px 16px rgba(59,130,246,0.35); }
.send-btn:disabled { opacity: 0.4; cursor: not-allowed; }

/* Markdown 渲染样式 */
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
</style>