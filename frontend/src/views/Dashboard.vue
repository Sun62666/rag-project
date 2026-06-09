<template>
  <div class="dashboard-view">
    <h2 class="dash-title"><i class="fa-solid fa-gauge-high"></i> 系统运行概览</h2>

    <div class="stat-cards">
      <div class="stat-card">
        <div class="stat-icon" style="background:rgba(59,130,246,0.12);color:#60a5fa"><i class="fa-solid fa-diagram-project"></i></div>
        <div class="stat-info">
          <div class="stat-value">{{ dashKGStats.total_nodes || '--' }}</div>
          <div class="stat-label">知识图谱实体</div>
        </div>
        <span :class="['stat-badge', dashKGStats.available ? 'online' : 'offline']">
          {{ dashKGStats.available ? '在线' : '离线' }}
        </span>
      </div>
      <div class="stat-card">
        <div class="stat-icon" style="background:rgba(139,92,246,0.12);color:#a78bfa"><i class="fa-solid fa-link"></i></div>
        <div class="stat-info">
          <div class="stat-value">{{ dashKGStats.total_relations || '--' }}</div>
          <div class="stat-label">知识图谱关系</div>
        </div>
      </div>
      <div class="stat-card">
        <div class="stat-icon" style="background:rgba(34,197,94,0.12);color:#4ade80"><i class="fa-solid fa-database"></i></div>
        <div class="stat-info">
          <div class="stat-value">{{ dashMilvusTotal || '--' }}</div>
          <div class="stat-label">向量知识库条目</div>
        </div>
      </div>
      <div class="stat-card">
        <div class="stat-icon" style="background:rgba(234,179,8,0.12);color:#facc15"><i class="fa-solid fa-file-lines"></i></div>
        <div class="stat-info">
          <div class="stat-value">{{ dashLogFiles.length }}</div>
          <div class="stat-label">日志文件</div>
        </div>
      </div>
    </div>

    <div class="dash-grid">
      <div class="dash-panel">
        <div class="panel-header"><strong>实体类型分布</strong></div>
        <div class="panel-body" v-if="dashKGStats.entity_types">
          <div v-for="(count, type) in dashKGStats.entity_types" :key="type" class="type-row">
            <span class="type-name">{{ type }}</span>
            <div class="type-bar-bg">
              <div class="type-bar-fill" :style="{ width: dashTypeBarWidth(count) + '%', background: dashTypeColor(type) }"></div>
            </div>
            <span class="type-count">{{ count }}</span>
          </div>
        </div>
        <div v-else class="panel-empty">暂无数据</div>
      </div>

      <div class="dash-panel">
        <div class="panel-header"><strong>快捷操作</strong></div>
        <div class="panel-body quick-grid">
          <button class="quick-action" @click="quickAsk('Linux服务器CPU使用率持续100%如何排查？')">
            <i class="fa-solid fa-microchip"></i> CPU 故障排查
          </button>
          <button class="quick-action" @click="quickAsk('磁盘空间不足怎么处理？')">
            <i class="fa-solid fa-hard-drive"></i> 磁盘空间不足
          </button>
          <button class="quick-action" @click="quickAsk('Docker容器无法启动怎么排查？')">
            <i class="fa-brands fa-docker"></i> Docker 容器排查
          </button>
          <button class="quick-action" @click="$router.push('/tools')">
            <i class="fa-solid fa-file-lines"></i> 查看系统日志
          </button>
          <button class="quick-action" @click="$router.push('/tools')">
            <i class="fa-solid fa-diagram-project"></i> 知识图谱分析
          </button>
          <button class="quick-action" @click="$router.push('/tools')">
            <i class="fa-solid fa-cloud-arrow-up"></i> 上传运维文档
          </button>
        </div>
      </div>
    </div>

    <div class="dash-panel">
      <div class="panel-header"><strong>最近会话</strong></div>
      <div class="panel-body" v-if="sessions.length > 0">
        <div v-for="sess in sessions.slice(0, 5)" :key="sess.session_id" class="session-row" @click="currentSessionId = sess.session_id; $router.push('/chat')">
          <i class="fa-regular fa-comment"></i>
          <span style="flex:1">{{ sess.title }}</span>
          <span class="session-time">{{ sess.created_at || '' }}</span>
        </div>
      </div>
      <div v-else class="panel-empty">暂无历史会话</div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, inject } from 'vue'
import { useRouter } from 'vue-router'
import { opsAPI } from '../api'

const router = useRouter()
const sessions = inject('sessions')
const currentSessionId = inject('currentSessionId')

const dashKGStats = ref({})
const dashMilvusTotal = ref(0)
const dashLogFiles = ref([])

const dashTypeBarWidth = (count) => {
  const max = Math.max(...Object.values(dashKGStats.value.entity_types || {}), 1)
  return Math.round((count / max) * 100)
}

const dashTypeColor = (type) => {
  const colors = { Component: '#3B82F6', Fault: '#EF4444', Command: '#22C55E', Config: '#A855F7', Metric: '#EAB308', Service: '#06B6D4' }
  return colors[type] || '#6B7280'
}

const quickAsk = (q) => {
  router.push({ path: '/chat', query: { q } })
}

onMounted(async () => {
  try {
    const [kgRes, milvusRes, logRes] = await Promise.all([
      opsAPI.kgStats(),
      opsAPI.knowledgeStats(),
      opsAPI.getLogFiles(),
    ])
    dashKGStats.value = kgRes.data
    let total = 0
    if (milvusRes.data.collections) {
      Object.values(milvusRes.data.collections).forEach(v => total += v)
    }
    dashMilvusTotal.value = total
    dashLogFiles.value = logRes.data.files || []
  } catch (e) { console.error('加载仪表盘失败:', e) }
})
</script>

<style scoped>
.dashboard-view { padding: 24px 28px; display: flex; flex-direction: column; gap: 20px; }
.dash-title { font-size: 20px; font-weight: 700; color: var(--text-primary); display: flex; align-items: center; gap: 10px; }

.stat-cards { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; }
.stat-card {
  background: var(--bg-secondary); border: 1px solid var(--border); border-radius: 12px;
  padding: 20px; display: flex; align-items: center; gap: 14px; position: relative;
}
.stat-icon {
  width: 44px; height: 44px; border-radius: 10px;
  display: flex; align-items: center; justify-content: center; font-size: 20px; flex-shrink: 0;
}
.stat-value { font-size: 26px; font-weight: 700; color: var(--text-primary); }
.stat-label { font-size: 13px; color: var(--text-muted); margin-top: 2px; }
.stat-badge {
  position: absolute; top: 12px; right: 12px; padding: 2px 8px; border-radius: 4px;
  font-size: 11px; font-weight: 600;
}
.stat-badge.online { background: rgba(34,197,94,0.12); color: #4ade80; }
.stat-badge.offline { background: rgba(239,68,68,0.12); color: #f87171; }

.dash-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.dash-panel {
  background: var(--bg-secondary); border: 1px solid var(--border); border-radius: 12px;
  overflow: hidden;
}
.panel-header {
  padding: 14px 20px; border-bottom: 1px solid var(--border);
  color: var(--text-primary); font-size: 14px;
}
.panel-body { padding: 16px 20px; }
.panel-empty { padding: 24px; text-align: center; color: var(--text-dim); font-size: 13px; }

.type-row { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
.type-name { width: 80px; font-size: 13px; color: var(--text-secondary); }
.type-bar-bg { flex: 1; height: 8px; background: var(--bg-tertiary); border-radius: 4px; overflow: hidden; }
.type-bar-fill { height: 100%; border-radius: 4px; transition: width 0.3s; }
.type-count { font-size: 13px; color: var(--text-muted); min-width: 30px; text-align: right; }

.quick-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.quick-action {
  padding: 12px; border-radius: 8px; border: 1px solid var(--border);
  background: var(--bg-primary); color: var(--text-secondary); font-size: 13px;
  cursor: pointer; transition: all 0.2s; display: flex; align-items: center; gap: 8px;
}
.quick-action:hover { border-color: var(--accent); color: var(--accent); background: rgba(59,130,246,0.06); }

.session-row {
  display: flex; align-items: center; gap: 10px; padding: 10px 0;
  border-bottom: 1px solid var(--border-light); cursor: pointer; color: var(--text-secondary);
  font-size: 13px; transition: all 0.15s;
}
.session-row:hover { color: var(--accent); }
.session-row:last-child { border-bottom: none; }
.session-time { color: var(--text-dim); font-size: 12px; }
</style>