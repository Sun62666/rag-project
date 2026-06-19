<template>
  <div class="tools-view">
    <div class="tools-tabs">
      <button :class="['tab-btn', { active: activeTab === 'logs' }]" @click="activeTab = 'logs'">
        <i class="fa-solid fa-file-lines"></i> 系统日志
      </button>
      <button :class="['tab-btn', { active: activeTab === 'docs' }]" @click="activeTab = 'docs'">
        <i class="fa-solid fa-cloud-arrow-up"></i> 文档上传
      </button>
      <button :class="['tab-btn', { active: activeTab === 'kg' }]" @click="activeTab = 'kg'">
        <i class="fa-solid fa-diagram-project"></i> 知识图谱
      </button>
    </div>

    <!-- 日志 -->
    <div v-if="activeTab === 'logs'" class="tab-content">
      <div class="toolbar">
        <select v-model="logFileName" class="tool-select" @change="loadLogs">
          <option v-for="f in logFileList" :key="f.name" :value="f.name">{{ f.name }}</option>
        </select>
        <select v-model="logLevel" class="tool-select" @change="loadLogs">
          <option value="all">全部</option>
          <option value="error">ERROR</option>
          <option value="warn">WARN</option>
          <option value="info">INFO</option>
        </select>
        <button class="tool-btn" @click="loadLogs"><i class="fa-solid fa-rotate"></i> 刷新</button>
      </div>
      <div class="log-table-wrap">
        <table class="log-table">
          <thead>
            <tr><th style="width:70px">行号</th><th style="width:80px">级别</th><th>内容</th></tr>
          </thead>
          <tbody>
            <tr v-for="row in logLines" :key="row.line_no">
              <td class="log-line-no">{{ row.line_no }}</td>
              <td><span :class="['level-tag', row.level]">{{ row.level || '-' }}</span></td>
              <td class="log-content">{{ row.content }}</td>
            </tr>
          </tbody>
        </table>
      </div>
      <div class="log-stats">
        共 {{ logTotalLines }} 行 | ERROR: {{ logErrorCount }} | WARN: {{ logWarnCount }}
      </div>
    </div>

    <!-- 文档上传 -->
    <div v-if="activeTab === 'docs'" class="tab-content">
      <div class="upload-zone" @dragover.prevent="dragover = true" @dragleave="dragover = false" @drop.prevent="handleDrop" :class="{ dragover }">
        <i class="fa-solid fa-cloud-arrow-up" style="font-size:40px;color:var(--accent);margin-bottom:12px"></i>
        <div style="color:var(--text-secondary);margin-bottom:4px">拖拽文件到此处，或 <em style="color:var(--accent);cursor:pointer" @click="$refs.fileInput.click()">点击上传</em></div>
        <div style="color:var(--text-dim);font-size:12px">支持 PDF / TXT / MD / DOCX 格式</div>
        <input ref="fileInput" type="file" accept=".pdf,.txt,.md,.docx" style="display:none" @change="handleFileInput" />
      </div>

      <div class="upload-target">
        <label style="color:var(--text-secondary);font-size:13px;white-space:nowrap">目标知识库：</label>
        <select v-model="uploadCollection" class="tool-select" style="flex:1">
          <option value="property_regulations">通用文档库 (property_regulations)</option>
          <option value="ops_knowledge_v2">运维知识库 (ops_knowledge_v2)</option>
        </select>
      </div>

      <div v-if="uploading" class="upload-progress">
        <div class="progress-bar"><div class="progress-fill" :style="{ width: uploadProgress + '%' }"></div></div>
      </div>
      <div v-if="uploadMessage" :class="['upload-msg', uploadStatus === 'ok' ? 'success' : 'error']">
        {{ uploadMessage }}
      </div>

      <div class="doc-list" v-if="uploadedDocs.length > 0">
        <div v-for="doc in uploadedDocs" :key="doc.name" class="doc-item">
          <i class="fa-regular fa-file-lines" style="color:var(--accent)"></i>
          <span style="flex:1">{{ doc.name }}</span>
          <span class="doc-size">{{ formatSize(doc.size) }}</span>
          <span class="doc-type">{{ doc.type }}</span>
          <button class="doc-del" @click="deleteDoc(doc.name)"><i class="fa-solid fa-trash-can"></i></button>
        </div>
      </div>

      <div class="knowledge-stats" v-if="knowledgeCollections">
        <h4 style="color:var(--text-primary);margin-bottom:10px">知识库统计</h4>
        <div v-for="(count, name) in knowledgeCollections" :key="name" class="ks-row">
          <span class="ks-name">{{ name }}</span>
          <span class="ks-count">{{ count }} 条</span>
        </div>
      </div>
    </div>

    <!-- 知识图谱 -->
    <div v-if="activeTab === 'kg'" class="tab-content">
      <div class="toolbar">
        <input v-model="kgSearchEntity" placeholder="搜索实体..." class="tool-input" @keyup.enter="searchKG" />
        <button class="tool-btn" @click="searchKG"><i class="fa-solid fa-magnifying-glass"></i> 搜索</button>
        <button class="tool-btn" @click="loadFullGraph"><i class="fa-solid fa-circle-nodes"></i> 加载全图</button>
        <button class="tool-btn accent" @click="$refs.kgFileInput.click()"><i class="fa-solid fa-file-import"></i> 导入数据</button>
        <input ref="kgFileInput" type="file" accept=".pdf,.txt,.md,.docx" style="display:none" @change="handleKGUpload" />
        <select v-model="kgMethod" class="tool-select" style="width:120px">
          <option value="hybrid">混合抽取</option>
          <option value="rule">规则抽取</option>
          <option value="spacy">spaCy</option>
          <option value="llm">LLM抽取</option>
        </select>
      </div>
      <div v-if="kgStats" class="kg-stats">
        <span class="kg-tag nodes">{{ kgStats.total_nodes || 0 }} 实体</span>
        <span class="kg-tag edges">{{ kgStats.total_relations || 0 }} 关系</span>
      </div>
      <div v-if="kgUploadMsg" :class="['upload-msg', kgUploadStatus === 'ok' ? 'success' : 'error']">
        {{ kgUploadMsg }}
      </div>
      <div v-if="kgLoading" style="text-align:center;padding:40px">
        <i class="fa-solid fa-spinner fa-spin" style="font-size:32px;color:var(--accent)"></i>
      </div>
      <template v-else>
        <div id="kg-network-container" class="kg-container"></div>
        <!-- 关系文字列表 -->
        <div v-if="kgRelations.length > 0" class="kg-relations-panel">
          <div class="kg-relations-header">
            <h4>关系列表</h4>
            <span class="kg-relations-count">{{ kgRelations.length }} 条关系</span>
          </div>
          <div class="kg-relations-list">
            <div v-for="(r, i) in kgRelations" :key="i" class="kg-relation-item">
              <span class="kg-entity" :class="r.source_type?.toLowerCase()">{{ r.source }}</span>
              <span class="kg-rel-arrow">
                <span class="kg-rel-label">{{ r.relation }}</span>
                <i class="fa-solid fa-arrow-right"></i>
              </span>
              <span class="kg-entity" :class="r.target_type?.toLowerCase()">{{ r.target }}</span>
            </div>
          </div>
        </div>
      </template>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, nextTick } from 'vue'
import { opsAPI } from '../api'
import { ElMessage, ElMessageBox } from 'element-plus'

const activeTab = ref('logs')

// ===== 日志 =====
const logLevel = ref('all')
const logLines = ref([])
const logErrorCount = ref(0)
const logWarnCount = ref(0)
const logTotalLines = ref(0)
const logFileList = ref([])
const logFileName = ref('smartops.log')

const loadLogs = async () => {
  try {
    const res = await opsAPI.getLogs({ lines: 200, level: logLevel.value, log_name: logFileName.value })
    logLines.value = res.data.lines
    logErrorCount.value = res.data.error_count
    logWarnCount.value = res.data.warn_count
    logTotalLines.value = res.data.total_lines
  } catch (e) { console.error('加载日志失败:', e) }
}

const loadLogFiles = async () => {
  try {
    const res = await opsAPI.getLogFiles()
    logFileList.value = res.data.files || []
  } catch (e) { console.error('加载日志文件列表失败:', e) }
}

// ===== 文档上传 =====
const uploadedDocs = ref([])
const uploading = ref(false)
const uploadProgress = ref(0)
const uploadMessage = ref('')
const uploadStatus = ref('ok')
const knowledgeCollections = ref(null)
const dragover = ref(false)
const uploadCollection = ref('property_regulations')

const handleFileInput = async (e) => {
  const file = e.target.files[0]
  if (file) await doUpload(file)
}

const handleDrop = async (e) => {
  dragover.value = false
  const file = e.dataTransfer.files[0]
  if (file) await doUpload(file)
}

const doUpload = async (file) => {
  uploading.value = true
  uploadMessage.value = ''
  const formData = new FormData()
  formData.append('file', file)
  formData.append('collection_name', uploadCollection.value)
  try {
    const res = await opsAPI.uploadDocument(formData)
    uploadMessage.value = res.data.message
    uploadStatus.value = res.data.status === 'ok' ? 'ok' : 'error'
    await loadUploadedDocs()
    await loadKnowledgeStats()
  } catch (e) {
    uploadMessage.value = '上传失败: ' + (e.response?.data?.detail || e.message)
    uploadStatus.value = 'error'
  } finally { uploading.value = false }
}

const loadUploadedDocs = async () => {
  try {
    const res = await opsAPI.listDocs()
    uploadedDocs.value = res.data.files || []
  } catch (e) { console.error('加载文档列表失败:', e) }
}

const loadKnowledgeStats = async () => {
  try {
    const res = await opsAPI.knowledgeStats()
    knowledgeCollections.value = res.data.collections && Object.keys(res.data.collections).length > 0
      ? res.data.collections : null
  } catch (e) { console.error('加载知识库统计失败:', e) }
}

const deleteDoc = async (filename) => {
  try {
    await ElMessageBox.confirm(`确定删除 ${filename} 吗？`, '提示')
    await opsAPI.deleteDoc(filename)
    await loadUploadedDocs()
  } catch { /* cancelled */ }
}

const formatSize = (bytes) => {
  if (bytes < 1024) return bytes + ' B'
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB'
  return (bytes / 1024 / 1024).toFixed(1) + ' MB'
}

// ===== 知识图谱 =====
const kgSearchEntity = ref('')
const kgStats = ref(null)
const kgLoading = ref(false)
const kgRelations = ref([])
const kgMethod = ref('hybrid')
const kgUploadMsg = ref('')
const kgUploadStatus = ref('ok')
let kgNetwork = null

const loadKGStats = async () => {
  try {
    const res = await opsAPI.kgStats()
    kgStats.value = res.data
  } catch (e) { console.error('加载图谱统计失败:', e) }
}

const loadFullGraph = async () => {
  kgLoading.value = true
  try {
    const [visRes, relRes] = await Promise.all([
      opsAPI.kgVis('', 2),
      opsAPI.kgRelations('', 2),
    ])
    await nextTick()
    renderKGNetwork(visRes.data.nodes || [], visRes.data.edges || [])
    kgRelations.value = relRes.data.relations || []
  } catch (e) { console.error('加载图谱失败:', e) }
  finally { kgLoading.value = false }
}

const searchKG = async () => {
  if (!kgSearchEntity.value.trim()) return
  kgLoading.value = true
  try {
    const [visRes, relRes] = await Promise.all([
      opsAPI.kgVis(kgSearchEntity.value, 1),
      opsAPI.kgRelations(kgSearchEntity.value, 1),
    ])
    await nextTick()
    renderKGNetwork(visRes.data.nodes || [], visRes.data.edges || [])
    kgRelations.value = relRes.data.relations || []
  } catch (e) { console.error('搜索图谱失败:', e) }
  finally { kgLoading.value = false }
}

const handleKGUpload = async (e) => {
  const file = e.target.files[0]
  if (!file) return
  kgUploadMsg.value = ''
  const formData = new FormData()
  formData.append('file', file)
  formData.append('method', kgMethod.value)
  try {
    const res = await opsAPI.kgUpload(formData)
    kgUploadMsg.value = res.data.message || `成功抽取 ${res.data.triples_extracted} 个三元组`
    kgUploadStatus.value = res.data.status === 'ok' ? 'ok' : 'error'
    await loadKGStats()
    await loadFullGraph()
  } catch (e) {
    kgUploadMsg.value = '上传失败: ' + (e.response?.data?.message || e.message)
    kgUploadStatus.value = 'error'
  }
  e.target.value = ''
}

const KG_COLORS = {
  Component: { background: '#3B82F6', border: '#2563EB' },
  Fault: { background: '#EF4444', border: '#DC2626' },
  Command: { background: '#22C55E', border: '#16A34A' },
  Config: { background: '#A855F7', border: '#9333EA' },
  Metric: { background: '#EAB308', border: '#CA8A04' },
  Service: { background: '#06B6D4', border: '#0891B2' },
  Entity: { background: '#6B7280', border: '#4B5563' },
}

const renderKGNetwork = async (nodes, edges) => {
  await nextTick()
  await new Promise(r => setTimeout(r, 200))
  if (typeof window.vis === 'undefined') {
    const script = document.createElement('script')
    script.src = 'https://unpkg.com/vis-network@9.1.6/standalone/umd/vis-network.min.js'
    script.onload = () => _doRender(nodes, edges)
    document.head.appendChild(script)
  } else { _doRender(nodes, edges) }
}

const _doRender = (nodes, edges) => {
  const container = document.getElementById('kg-network-container')
  if (!container) return
  const visNodes = nodes.map(n => {
    const colors = KG_COLORS[n.group] || KG_COLORS.Entity
    return { id: n.id, label: n.label, group: n.group, shape: 'dot', size: 16, color: colors, font: { color: '#e0e0e0', size: 13, strokeWidth: 2, strokeColor: 'rgba(15,23,42,0.8)' } }
  })
  const visEdges = edges.map(e => ({
    id: e.id, from: e.from, to: e.to, label: e.label, arrows: 'to',
    color: { color: 'rgba(255,255,255,0.2)', highlight: '#3B82F6' },
    font: { color: '#e0e0e0', size: 12, strokeWidth: 3, strokeColor: 'rgba(15,23,42,0.9)' },
    smooth: { type: 'curvedCW', roundness: 0.15 },
  }))
  if (kgNetwork) { kgNetwork.destroy(); kgNetwork = null }
  kgNetwork = new window.vis.Network(container,
    { nodes: new window.vis.DataSet(visNodes), edges: new window.vis.DataSet(visEdges) },
    { physics: { solver: 'forceAtlas2Based', forceAtlas2Based: { gravitationalConstant: -60, centralGravity: 0.01, springLength: 120 } } }
  )
}

onMounted(async () => {
  await Promise.all([loadLogs(), loadLogFiles(), loadUploadedDocs(), loadKnowledgeStats(), loadKGStats()])
})
</script>

<style scoped>
.tools-view { padding: 24px 28px; display: flex; flex-direction: column; gap: 16px; }

.tools-tabs { display: flex; gap: 4px; }
.tab-btn {
  padding: 9px 18px; border-radius: 8px; border: 1px solid var(--border);
  background: transparent; color: var(--text-muted); font-size: 13px;
  cursor: pointer; transition: all 0.15s; display: flex; align-items: center; gap: 6px;
}
.tab-btn.active { background: rgba(59,130,246,0.12); color: var(--accent); border-color: rgba(59,130,246,0.3); }
.tab-btn:hover:not(.active) { background: rgba(255,255,255,0.04); color: var(--text-secondary); }

.tab-content { display: flex; flex-direction: column; gap: 12px; }

.toolbar { display: flex; gap: 8px; align-items: center; }
.tool-select {
  background: var(--bg-primary); border: 1px solid var(--border); border-radius: 6px;
  padding: 7px 12px; color: var(--text-primary); font-size: 13px; outline: none;
}
.tool-select:focus { border-color: var(--accent); }
.tool-input {
  background: var(--bg-primary); border: 1px solid var(--border); border-radius: 6px;
  padding: 7px 12px; color: var(--text-primary); font-size: 13px; outline: none; width: 200px;
}
.tool-input::placeholder { color: var(--text-dim); }
.tool-input:focus { border-color: var(--accent); }
.tool-btn {
  padding: 7px 14px; border-radius: 6px; border: 1px solid var(--border);
  background: transparent; color: var(--text-secondary); font-size: 13px;
  cursor: pointer; transition: all 0.15s; display: flex; align-items: center; gap: 6px;
}
.tool-btn:hover { border-color: var(--accent); color: var(--accent); }

/* 日志表格 */
.log-table-wrap { overflow-x: auto; border: 1px solid var(--border); border-radius: 8px; }
.log-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.log-table th { background: var(--bg-tertiary); color: var(--text-primary); padding: 10px 12px; text-align: left; font-weight: 600; border-bottom: 1px solid var(--border); }
.log-table td { padding: 8px 12px; border-bottom: 1px solid var(--border-light); color: var(--text-secondary); }
.log-table tr:hover td { background: rgba(59,130,246,0.04); }
.log-line-no { color: var(--text-dim); font-family: monospace; }
.log-content { font-family: 'Consolas', monospace; font-size: 12px; }
.level-tag { padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; }
.level-tag.error { background: rgba(239,68,68,0.12); color: #f87171; }
.level-tag.warn { background: rgba(234,179,8,0.12); color: #facc15; }
.level-tag.info { background: rgba(59,130,246,0.12); color: #60a5fa; }
.log-stats { font-size: 12px; color: var(--text-dim); }

/* 上传 */
.upload-zone {
  border: 2px dashed var(--border); border-radius: 12px; padding: 40px;
  text-align: center; transition: all 0.2s; cursor: pointer;
}
.upload-zone:hover, .upload-zone.dragover { border-color: var(--accent); background: rgba(59,130,246,0.04); }
.upload-target { display: flex; align-items: center; gap: 10px; }
.upload-progress { margin: 8px 0; }
.progress-bar { height: 4px; background: var(--bg-tertiary); border-radius: 2px; overflow: hidden; }
.progress-fill { height: 100%; background: var(--accent); border-radius: 2px; transition: width 0.3s; }
.upload-msg { padding: 10px 14px; border-radius: 8px; font-size: 13px; }
.upload-msg.success { background: rgba(34,197,94,0.1); color: #4ade80; border: 1px solid rgba(34,197,94,0.2); }
.upload-msg.error { background: rgba(239,68,68,0.1); color: #f87171; border: 1px solid rgba(239,68,68,0.2); }

.doc-list { display: flex; flex-direction: column; gap: 4px; }
.doc-item {
  display: flex; align-items: center; gap: 10px; padding: 10px 14px;
  background: var(--bg-secondary); border: 1px solid var(--border); border-radius: 8px;
  font-size: 13px; color: var(--text-secondary);
}
.doc-size { color: var(--text-dim); font-size: 12px; }
.doc-type { color: var(--text-dim); font-size: 12px; }
.doc-del {
  background: none; border: none; color: var(--text-dim); cursor: pointer;
  padding: 4px; border-radius: 4px; transition: all 0.15s;
}
.doc-del:hover { color: var(--danger); background: rgba(239,68,68,0.1); }

.knowledge-stats { margin-top: 16px; padding: 16px; background: var(--bg-secondary); border: 1px solid var(--border); border-radius: 8px; }
.ks-row { display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid var(--border-light); font-size: 13px; }
.ks-name { color: var(--text-secondary); }
.ks-count { color: var(--accent); font-weight: 600; }

/* 知识图谱 */
.kg-stats { display: flex; gap: 8px; }
.kg-tag { padding: 4px 10px; border-radius: 4px; font-size: 12px; font-weight: 600; }
.kg-tag.nodes { background: rgba(34,197,94,0.12); color: #4ade80; }
.kg-tag.edges { background: rgba(234,179,8,0.12); color: #facc15; }
.kg-container { height: 500px; border: 1px solid var(--border); border-radius: 8px; background: #0f172a; }

/* 导入数据按钮 */
.tool-btn.accent {
  background: rgba(59,130,246,0.12); color: var(--accent); border-color: rgba(59,130,246,0.3);
}
.tool-btn.accent:hover { background: rgba(59,130,246,0.2); }

/* 关系文字列表 */
.kg-relations-panel {
  margin-top: 16px; border: 1px solid var(--border); border-radius: 8px;
  background: var(--bg-secondary); overflow: hidden;
}
.kg-relations-header {
  display: flex; justify-content: space-between; align-items: center;
  padding: 12px 16px; border-bottom: 1px solid var(--border);
}
.kg-relations-header h4 { color: var(--text-primary); margin: 0; font-size: 14px; }
.kg-relations-count { color: var(--text-dim); font-size: 12px; }
.kg-relations-list {
  max-height: 400px; overflow-y: auto; padding: 8px 0;
}
.kg-relation-item {
  display: flex; align-items: center; gap: 10px;
  padding: 8px 16px; font-size: 13px; transition: background 0.1s;
}
.kg-relation-item:hover { background: rgba(59,130,246,0.04); }
.kg-entity {
  padding: 3px 10px; border-radius: 4px; font-weight: 500; white-space: nowrap;
}
.kg-entity.component { background: rgba(59,130,246,0.15); color: #60a5fa; }
.kg-entity.fault { background: rgba(239,68,68,0.15); color: #f87171; }
.kg-entity.command { background: rgba(34,197,94,0.15); color: #4ade80; }
.kg-entity.config { background: rgba(168,85,247,0.15); color: #c084fc; }
.kg-entity.metric { background: rgba(234,179,8,0.15); color: #facc15; }
.kg-entity.service { background: rgba(6,182,212,0.15); color: #22d3ee; }
.kg-entity.protocol { background: rgba(244,114,182,0.15); color: #f472b6; }
.kg-entity.entity { background: rgba(107,114,128,0.15); color: #9ca3af; }
.kg-rel-arrow {
  display: flex; align-items: center; gap: 6px; color: var(--text-dim); flex-shrink: 0;
}
.kg-rel-label {
  padding: 2px 8px; border-radius: 3px; font-size: 11px;
  background: rgba(255,255,255,0.06); border: 1px solid var(--border);
}
</style>