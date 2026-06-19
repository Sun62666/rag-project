<template>
  <div class="evaluate-view">
    <!-- 标题栏 -->
    <div class="eval-header">
      <div>
        <h2 style="margin:0;color:var(--text-primary)">检索效果评估</h2>
        <span style="color:var(--text-muted);font-size:13px">多策略评估: Precision / Recall / MRR / NDCG / F1</span>
      </div>
      <div class="eval-controls">
        <span style="font-size:13px;color:var(--text-muted)">相似度阈值:</span>
        <input type="number" v-model.number="threshold" min="0" max="1" step="0.05" class="threshold-input" />
        <span style="font-size:13px;color:var(--text-muted)">K值:</span>
        <input type="number" v-model.number="kValue" min="1" max="20" class="threshold-input" style="width:50px" />
        <button class="eval-run-btn" :disabled="evaluating" @click="runEvaluation">
          <i class="fa-solid fa-chart-line"></i> {{ evaluating ? '评估中...' : '运行评估' }}
        </button>
      </div>
    </div>

    <!-- Tab 切换 -->
    <div class="tab-bar">
      <button :class="['tab-btn', { active: activeTab === 'preset' }]" @click="activeTab = 'preset'">
        <i class="fa-solid fa-list-check"></i> 预定义问题
      </button>
      <button :class="['tab-btn', { active: activeTab === 'custom' }]" @click="activeTab = 'custom'">
        <i class="fa-solid fa-pen-to-square"></i> 自定义问题
      </button>
    </div>

    <!-- 预定义问题面板 -->
    <div v-show="activeTab === 'preset'" class="preset-panel">
      <!-- 领域筛选 -->
      <div class="domain-filter">
        <button :class="['domain-btn', { active: selectedDomain === 'all' }]" @click="switchDomain('all')">
          全部 ({{ allQuestions.length }})
        </button>
        <button :class="['domain-btn', { active: selectedDomain === 'ops' }]" @click="switchDomain('ops')">
          <i class="fa-solid fa-server"></i> 运维 ({{ opsQuestions.length }})
        </button>
        <button :class="['domain-btn', { active: selectedDomain === 'document' }]" @click="switchDomain('document')">
          <i class="fa-solid fa-file-lines"></i> 物业文档 ({{ docQuestions.length }})
        </button>
      </div>

      <!-- 问题列表（可勾选） -->
      <div class="question-list">
        <div class="select-all-row">
          <label class="checkbox-label">
            <input type="checkbox" :checked="allSelected" @change="toggleSelectAll" />
            <span>全选 / 取消全选</span>
          </label>
          <span class="selected-count">已选 {{ selectedIndices.length }} / {{ currentQuestions.length }} 题</span>
        </div>
        <div v-for="(eq, i) in currentQuestions" :key="eq.index" :class="['question-item', { selected: isSelected(eq.index) }]">
          <label class="checkbox-label">
            <input type="checkbox" :checked="isSelected(eq.index)" @change="toggleSelect(eq.index)" />
            <span class="q-domain-tag" :class="eq.domain">{{ eq.domain === 'ops' ? '运维' : '物业' }}</span>
            <span class="q-text">{{ eq.question }}</span>
          </label>
        </div>
      </div>
    </div>

    <!-- 自定义问题面板 -->
    <div v-show="activeTab === 'custom'" class="custom-panel">
      <div class="custom-form">
        <div class="form-group">
          <label>问题</label>
          <textarea v-model="customQuestion" placeholder="输入要评估的问题..." rows="2" class="form-input"></textarea>
        </div>
        <div class="form-group">
          <label>标准答案（Ground Truth）</label>
          <textarea v-model="customGroundTruth" placeholder="输入标准答案，用于对比检索结果..." rows="4" class="form-input"></textarea>
        </div>
        <button class="eval-run-btn" :disabled="evaluatingCustom || !customQuestion.trim() || !customGroundTruth.trim()" @click="runCustomEval">
          <i class="fa-solid fa-play"></i> {{ evaluatingCustom ? '评估中...' : '评估自定义问题' }}
        </button>
      </div>
    </div>

    <!-- 评估结果 -->
    <div v-if="evalResult" class="eval-results">
      <!-- 汇总指标卡片 -->
      <div class="metrics-grid">
        <div v-for="metric in metricsCards" :key="metric.key" :class="['metric-card', metric.level]">
          <div class="metric-icon"><i :class="metric.icon"></i></div>
          <div class="metric-body">
            <div class="metric-value">{{ metric.value }}</div>
            <div class="metric-label">{{ metric.label }}</div>
          </div>
        </div>
      </div>

      <!-- 可视化图表：各问题指标对比 -->
      <div class="chart-section" v-if="evalResult.results && evalResult.results.length > 0">
        <h4 style="color:var(--text-primary);margin:0 0 12px 0">
          <i class="fa-solid fa-chart-bar"></i> 各问题指标对比
        </h4>
        <div class="bar-chart">
          <div v-for="(r, i) in evalResult.results" :key="i" class="bar-group">
            <div class="bar-label" :title="r.question">Q{{ i + 1 }}</div>
            <div class="bar-track">
              <div class="bar-fill precision" :style="{ width: (r.precision_at_k * 100) + '%' }" :title="'Precision: ' + r.precision_at_k"></div>
              <div class="bar-fill recall" :style="{ width: (r.recall_at_k * 100) + '%' }" :title="'Recall: ' + r.recall_at_k"></div>
              <div class="bar-fill mrr" :style="{ width: (r.mrr * 100) + '%' }" :title="'MRR: ' + r.mrr"></div>
              <div class="bar-fill ndcg" :style="{ width: (r.ndcg_at_k * 100) + '%' }" :title="'NDCG: ' + r.ndcg_at_k"></div>
            </div>
          </div>
        </div>
        <div class="chart-legend">
          <span class="legend-item"><span class="legend-color precision"></span> Precision@{{ evalResult.summary?.k || kValue }}</span>
          <span class="legend-item"><span class="legend-color recall"></span> Recall@{{ evalResult.summary?.k || kValue }}</span>
          <span class="legend-item"><span class="legend-color mrr"></span> MRR</span>
          <span class="legend-item"><span class="legend-color ndcg"></span> NDCG@{{ evalResult.summary?.k || kValue }}</span>
        </div>
      </div>

      <!-- 雷达图：平均指标 -->
      <div class="chart-section" v-if="evalResult.summary">
        <h4 style="color:var(--text-primary);margin:0 0 12px 0">
          <i class="fa-solid fa-chart-pie"></i> 平均指标雷达图
        </h4>
        <div class="radar-chart">
          <svg viewBox="0 0 200 200" class="radar-svg">
            <!-- 网格 -->
            <polygon points="100,20 168,60 168,140 100,180 32,140 32,60" fill="none" stroke="var(--border)" stroke-width="0.5" />
            <polygon points="100,40 151,70 151,130 100,160 49,130 49,70" fill="none" stroke="var(--border)" stroke-width="0.5" />
            <polygon points="100,60 134,80 134,120 100,140 66,120 66,80" fill="none" stroke="var(--border)" stroke-width="0.5" />
            <polygon points="100,80 117,90 117,110 100,120 83,110 83,90" fill="none" stroke="var(--border)" stroke-width="0.5" />
            <!-- 轴线 -->
            <line x1="100" y1="100" x2="100" y2="20" stroke="var(--border)" stroke-width="0.5" />
            <line x1="100" y1="100" x2="168" y2="60" stroke="var(--border)" stroke-width="0.5" />
            <line x1="100" y1="100" x2="168" y2="140" stroke="var(--border)" stroke-width="0.5" />
            <line x1="100" y1="100" x2="100" y2="180" stroke="var(--border)" stroke-width="0.5" />
            <line x1="100" y1="100" x2="32" y2="140" stroke="var(--border)" stroke-width="0.5" />
            <line x1="100" y1="100" x2="32" y2="60" stroke="var(--border)" stroke-width="0.5" />
            <!-- 数据多边形 -->
            <polygon :points="radarPoints" fill="rgba(59,130,246,0.15)" stroke="#3B82F6" stroke-width="1.5" />
            <!-- 标签 -->
            <text x="100" y="14" text-anchor="middle" fill="var(--text-muted)" font-size="8">Precision</text>
            <text x="174" y="58" text-anchor="start" fill="var(--text-muted)" font-size="8">Recall</text>
            <text x="174" y="146" text-anchor="start" fill="var(--text-muted)" font-size="8">F1</text>
            <text x="100" y="194" text-anchor="middle" fill="var(--text-muted)" font-size="8">NDCG</text>
            <text x="26" y="146" text-anchor="end" fill="var(--text-muted)" font-size="8">MRR</text>
            <text x="26" y="58" text-anchor="end" fill="var(--text-muted)" font-size="8">-</text>
          </svg>
        </div>
      </div>

      <!-- 详细结果列表 -->
      <h4 style="color:var(--text-primary);margin:0">详细结果</h4>
      <div v-for="(result, i) in (evalResult.results || (evalResult.result ? [evalResult.result] : []))" :key="i" class="result-card">
        <div class="result-header" @click="toggleExpand(i)" style="cursor:pointer">
          <div style="display:flex;align-items:center;gap:8px;flex:1;min-width:0">
            <i :class="expandedCards[i] ? 'fa-solid fa-chevron-down' : 'fa-solid fa-chevron-right'" style="color:var(--text-dim);font-size:12px;width:16px"></i>
            <span class="q-domain-tag" :class="result.domain">{{ result.domain === 'ops' ? '运维' : result.domain === 'document' ? '物业' : '自定义' }}</span>
            <strong style="color:var(--text-primary);overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{{ result.question }}</strong>
          </div>
          <div class="result-metrics">
            <span class="mini-metric" title="Precision">P: {{ result.precision_at_k }}</span>
            <span class="mini-metric" title="Recall">R: {{ result.recall_at_k }}</span>
            <span class="mini-metric" title="MRR">MRR: {{ result.mrr }}</span>
            <span class="mini-metric" title="NDCG">NDCG: {{ result.ndcg_at_k }}</span>
            <span class="mini-metric" title="F1">F1: {{ result.f1_at_k }}</span>
          </div>
        </div>
        <div v-show="expandedCards[i]" class="chunks-list">
          <div class="tp-info">
            <i class="fa-solid fa-circle-check" style="color:#4ade80"></i>
            命中 {{ result.tp_count }} / {{ result.total_chunks || 0 }} 个相关切片
          </div>
          <div v-for="(chunk, ci) in (result.chunks || [])" :key="ci" class="chunk-row" @click="toggleChunk(i, ci)" style="cursor:pointer">
            <div class="chunk-top">
              <span class="chunk-rank">#{{ ci + 1 }}</span>
              <span class="chunk-id">{{ chunk.chunk_id }}</span>
              <span :class="['chunk-sim', chunk.similarity >= threshold ? 'pass' : 'fail']">{{ chunk.similarity }}</span>
              <span class="chunk-rel">{{ chunk.relevant ? '✓' : '✗' }}</span>
              <i :class="expandedChunks[`${i}-${ci}`] ? 'fa-solid fa-chevron-up' : 'fa-solid fa-chevron-down'" style="color:var(--text-dim);font-size:11px;margin-left:auto"></i>
            </div>
            <div :class="['chunk-content', { expanded: expandedChunks[`${i}-${ci}`] }]">
              {{ chunk.content }}
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- 空状态 -->
    <div v-else class="eval-empty">
      <i class="fa-solid fa-chart-line" style="font-size:48px;color:var(--text-dim);margin-bottom:12px"></i>
      <div>选择评估问题或输入自定义问题，点击"运行评估"开始</div>
      <div style="font-size:13px;color:var(--text-dim);margin-top:4px">请先确保知识库中已有文档数据</div>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, computed, onMounted } from 'vue'
import { evaluateAPI } from '../api'
import { ElMessage } from 'element-plus'

defineOptions({ name: 'Evaluate' })

// ===== 参数 =====
const threshold = ref(0.7)
const kValue = ref(10)
const evaluating = ref(false)
const evaluatingCustom = ref(false)
const evalResult = ref(null)
const expandedCards = reactive({})
const expandedChunks = reactive({})

// ===== Tab =====
const activeTab = ref('preset')

// ===== 预定义问题 =====
const allQuestions = ref([])
const opsQuestions = ref([])
const docQuestions = ref([])
const selectedDomain = ref('all')
const selectedIndices = ref([])

const currentQuestions = computed(() => {
  if (selectedDomain.value === 'ops') return opsQuestions.value
  if (selectedDomain.value === 'document') return docQuestions.value
  return allQuestions.value
})

const allSelected = computed(() => {
  return currentQuestions.value.length > 0 && currentQuestions.value.every(q => selectedIndices.value.includes(q.index))
})

const switchDomain = (domain) => {
  selectedDomain.value = domain
}

const isSelected = (idx) => selectedIndices.value.includes(idx)

const toggleSelect = (idx) => {
  const pos = selectedIndices.value.indexOf(idx)
  if (pos >= 0) {
    selectedIndices.value.splice(pos, 1)
  } else {
    selectedIndices.value.push(idx)
  }
}

const toggleSelectAll = () => {
  if (allSelected.value) {
    // 取消当前领域的所有选择
    currentQuestions.value.forEach(q => {
      const pos = selectedIndices.value.indexOf(q.index)
      if (pos >= 0) selectedIndices.value.splice(pos, 1)
    })
  } else {
    // 选中当前领域的所有
    currentQuestions.value.forEach(q => {
      if (!selectedIndices.value.includes(q.index)) {
        selectedIndices.value.push(q.index)
      }
    })
  }
}

// ===== 自定义问题 =====
const customQuestion = ref('')
const customGroundTruth = ref('')

// ===== 指标卡片 =====
const metricsCards = computed(() => {
  const s = evalResult.value?.summary
  if (!s) return []
  const getLevel = (v) => v >= 0.7 ? 'good' : v >= 0.4 ? 'warn' : 'bad'
  return [
    { key: 'precision', label: `Precision@${s.k}`, value: s.avg_precision_at_k, icon: 'fa-solid fa-bullseye', level: getLevel(s.avg_precision_at_k) },
    { key: 'recall', label: `Recall@${s.k}`, value: s.avg_recall_at_k, icon: 'fa-solid fa-magnifying-glass', level: getLevel(s.avg_recall_at_k) },
    { key: 'mrr', label: 'MRR', value: s.avg_mrr, icon: 'fa-solid fa-arrow-down-short-wide', level: getLevel(s.avg_mrr) },
    { key: 'ndcg', label: `NDCG@${s.k}`, value: s.avg_ndcg_at_k, icon: 'fa-solid fa-chart-line', level: getLevel(s.avg_ndcg_at_k) },
    { key: 'f1', label: `F1@${s.k}`, value: s.avg_f1_at_k, icon: 'fa-solid fa-scale-balanced', level: getLevel(s.avg_f1_at_k) },
  ]
})

// ===== 雷达图数据点 =====
const radarPoints = computed(() => {
  const s = evalResult.value?.summary
  if (!s) return ''
  const cx = 100, cy = 100, r = 80
  const vals = [s.avg_precision_at_k, s.avg_recall_at_k, s.avg_f1_at_k, s.avg_ndcg_at_k, s.avg_mrr]
  const angles = [-90, -30, 30, 90, 150] // 5个方向
  const pts = vals.map((v, i) => {
    const rad = (angles[i] * Math.PI) / 180
    const x = cx + r * v * Math.cos(rad)
    const y = cy + r * v * Math.sin(rad)
    return `${x.toFixed(1)},${y.toFixed(1)}`
  })
  return pts.join(' ')
})

// ===== 加载问题 =====
const loadQuestions = async () => {
  try {
    const [allRes, opsRes, docRes] = await Promise.all([
      evaluateAPI.getQuestions('all'),
      evaluateAPI.getQuestions('ops'),
      evaluateAPI.getQuestions('document'),
    ])
    allQuestions.value = allRes.data.questions || []
    opsQuestions.value = opsRes.data.questions || []
    docQuestions.value = docRes.data.questions || []
    // 默认全选
    selectedIndices.value = allQuestions.value.map(q => q.index)
  } catch (e) {
    console.error('加载评估问题失败:', e)
  }
}

// ===== 运行评估 =====
const runEvaluation = async () => {
  if (selectedIndices.value.length === 0) {
    ElMessage.warning('请至少选择一个问题')
    return
  }
  evaluating.value = true
  evalResult.value = null
  try {
    const res = await evaluateAPI.run({
      threshold: threshold.value,
      k: kValue.value,
      domain: selectedDomain.value,
      selected_indices: selectedIndices.value,
    })
    if (res.data.status === 'ok') {
      evalResult.value = res.data
      if (res.data.results) {
        res.data.results.forEach((_, i) => { expandedCards[i] = true })
      }
      ElMessage.success(`评估完成，共 ${res.data.summary?.valid_questions || 0} 个有效问题`)
    } else {
      ElMessage.error(res.data.message || '评估失败')
    }
  } catch (e) {
    ElMessage.error('评估请求失败: ' + (e.response?.data?.detail || e.message))
  } finally {
    evaluating.value = false
  }
}

// ===== 自定义评估 =====
const runCustomEval = async () => {
  evaluatingCustom.value = true
  evalResult.value = null
  try {
    const res = await evaluateAPI.custom({
      question: customQuestion.value,
      ground_truth: customGroundTruth.value,
      threshold: threshold.value,
      k: kValue.value,
    })
    if (res.data.status === 'ok') {
      evalResult.value = { result: res.data.result, results: null, summary: null }
      expandedCards[0] = true
      ElMessage.success('自定义评估完成')
    } else {
      ElMessage.error(res.data.message || '评估失败')
    }
  } catch (e) {
    ElMessage.error('评估请求失败: ' + (e.response?.data?.detail || e.message))
  } finally {
    evaluatingCustom.value = false
  }
}

// ===== 展开/折叠 =====
const toggleExpand = (i) => {
  expandedCards[i] = !expandedCards[i]
}

const toggleChunk = (i, ci) => {
  const key = `${i}-${ci}`
  expandedChunks[key] = !expandedChunks[key]
}

onMounted(() => { loadQuestions() })
</script>

<style scoped>
.evaluate-view { padding: 24px 28px; display: flex; flex-direction: column; gap: 20px; max-height: 100%; overflow-y: auto; }

.eval-header { display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px; }
.eval-controls { display: flex; align-items: center; gap: 8px; }
.threshold-input {
  width: 70px; background: var(--bg-primary); border: 1px solid var(--border);
  border-radius: 6px; padding: 6px 10px; color: var(--text-primary); font-size: 13px; outline: none;
}
.threshold-input:focus { border-color: var(--accent); }
.eval-run-btn {
  padding: 8px 16px; border-radius: 8px; border: none;
  background: linear-gradient(135deg, #3B82F6, #6366F1); color: #fff;
  font-size: 13px; cursor: pointer; transition: all 0.2s;
  display: flex; align-items: center; gap: 6px; white-space: nowrap;
}
.eval-run-btn:hover:not(:disabled) { transform: translateY(-1px); box-shadow: 0 4px 16px rgba(59,130,246,0.35); }
.eval-run-btn:disabled { opacity: 0.5; cursor: not-allowed; }

/* Tab */
.tab-bar { display: flex; gap: 4px; border-bottom: 2px solid var(--border); }
.tab-btn {
  padding: 8px 20px; border: none; background: none; color: var(--text-muted);
  font-size: 14px; cursor: pointer; border-bottom: 2px solid transparent; margin-bottom: -2px;
  transition: all 0.2s; display: flex; align-items: center; gap: 6px;
}
.tab-btn:hover { color: var(--text-secondary); }
.tab-btn.active { color: var(--accent); border-bottom-color: var(--accent); }

/* 领域筛选 */
.domain-filter { display: flex; gap: 8px; flex-wrap: wrap; }
.domain-btn {
  padding: 6px 14px; border-radius: 20px; border: 1px solid var(--border);
  background: var(--bg-secondary); color: var(--text-muted); font-size: 13px;
  cursor: pointer; transition: all 0.2s; display: flex; align-items: center; gap: 6px;
}
.domain-btn:hover { border-color: var(--accent); color: var(--accent); }
.domain-btn.active { background: rgba(59,130,246,0.15); border-color: var(--accent); color: var(--accent); }

/* 问题列表 */
.question-list { background: var(--bg-secondary); border: 1px solid var(--border); border-radius: 12px; padding: 12px 16px; }
.select-all-row { display: flex; justify-content: space-between; align-items: center; padding-bottom: 8px; border-bottom: 1px solid var(--border); margin-bottom: 8px; }
.selected-count { font-size: 13px; color: var(--text-muted); }
.checkbox-label { display: flex; align-items: center; gap: 8px; cursor: pointer; font-size: 13px; color: var(--text-secondary); }
.checkbox-label input[type="checkbox"] { width: 16px; height: 16px; accent-color: var(--accent); cursor: pointer; }
.question-item { padding: 8px 12px; border-radius: 6px; margin-bottom: 4px; transition: background 0.2s; }
.question-item:hover { background: rgba(59,130,246,0.06); }
.question-item.selected { background: rgba(59,130,246,0.08); }
.q-domain-tag {
  display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; white-space: nowrap;
}
.q-domain-tag.ops { background: rgba(59,130,246,0.15); color: #60a5fa; }
.q-domain-tag.document { background: rgba(168,85,247,0.15); color: #c084fc; }
.q-domain-tag.custom { background: rgba(34,197,94,0.15); color: #4ade80; }
.q-text { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

/* 自定义表单 */
.custom-panel { background: var(--bg-secondary); border: 1px solid var(--border); border-radius: 12px; padding: 20px; }
.custom-form { display: flex; flex-direction: column; gap: 16px; }
.form-group { display: flex; flex-direction: column; gap: 6px; }
.form-group label { font-size: 13px; color: var(--text-secondary); font-weight: 600; }
.form-input {
  background: var(--bg-primary); border: 1px solid var(--border); border-radius: 8px;
  padding: 10px 14px; color: var(--text-primary); font-size: 14px; font-family: inherit;
  outline: none; resize: vertical;
}
.form-input:focus { border-color: var(--accent); }

/* 指标卡片 */
.metrics-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 12px; }
.metric-card {
  display: flex; align-items: center; gap: 12px; padding: 16px 20px; border-radius: 12px;
  border: 1px solid var(--border); border-left: 4px solid;
}
.metric-card.good { border-left-color: #22C55E; background: rgba(34,197,94,0.06); }
.metric-card.warn { border-left-color: #EAB308; background: rgba(234,179,8,0.06); }
.metric-card.bad { border-left-color: #EF4444; background: rgba(239,68,68,0.06); }
.metric-icon { font-size: 24px; color: var(--accent); }
.metric-value { font-size: 22px; font-weight: 700; color: var(--text-primary); }
.metric-label { font-size: 12px; color: var(--text-muted); margin-top: 2px; }

/* 图表区域 */
.chart-section { background: var(--bg-secondary); border: 1px solid var(--border); border-radius: 12px; padding: 16px 20px; }

/* 柱状图 */
.bar-chart { display: flex; flex-direction: column; gap: 6px; }
.bar-group { display: flex; align-items: center; gap: 10px; }
.bar-label { width: 36px; font-size: 12px; color: var(--text-muted); text-align: right; flex-shrink: 0; }
.bar-track { flex: 1; display: flex; flex-direction: column; gap: 2px; height: 24px; }
.bar-fill { height: 5px; border-radius: 2px; transition: width 0.5s ease; min-width: 0; }
.bar-fill.precision { background: #3B82F6; }
.bar-fill.recall { background: #22C55E; }
.bar-fill.mrr { background: #F59E0B; }
.bar-fill.ndcg { background: #A855F7; }
.chart-legend { display: flex; gap: 16px; margin-top: 12px; flex-wrap: wrap; }
.legend-item { display: flex; align-items: center; gap: 6px; font-size: 12px; color: var(--text-muted); }
.legend-color { width: 12px; height: 12px; border-radius: 3px; display: inline-block; }
.legend-color.precision { background: #3B82F6; }
.legend-color.recall { background: #22C55E; }
.legend-color.mrr { background: #F59E0B; }
.legend-color.ndcg { background: #A855F7; }

/* 雷达图 */
.radar-chart { display: flex; justify-content: center; }
.radar-svg { width: 260px; height: 260px; }

/* 结果卡片 */
.eval-results { display: flex; flex-direction: column; gap: 16px; }
.result-card { background: var(--bg-secondary); border: 1px solid var(--border); border-left: 3px solid var(--accent); border-radius: 10px; padding: 14px 20px; }
.result-header { display: flex; justify-content: space-between; align-items: center; gap: 12px; }
.result-metrics { display: flex; gap: 6px; flex-shrink: 0; }
.mini-metric { font-size: 11px; padding: 3px 8px; border-radius: 4px; background: var(--bg-primary); color: var(--text-muted); white-space: nowrap; }

.chunks-list { display: flex; flex-direction: column; gap: 6px; margin-top: 12px; }
.tp-info { font-size: 13px; color: var(--text-secondary); padding: 6px 0; }
.chunk-row { padding: 10px 14px; background: var(--bg-primary); border-radius: 8px; border: 1px solid var(--border-light); transition: border-color 0.2s; }
.chunk-row:hover { border-color: var(--accent); }
.chunk-top { display: flex; align-items: center; gap: 10px; margin-bottom: 6px; }
.chunk-rank { color: var(--accent); font-weight: 700; font-size: 13px; min-width: 28px; }
.chunk-id { color: var(--text-dim); font-size: 12px; min-width: 80px; }
.chunk-sim { font-weight: 600; min-width: 50px; text-align: right; font-size: 13px; }
.chunk-sim.pass { color: #4ade80; }
.chunk-sim.fail { color: #f87171; }
.chunk-rel { min-width: 20px; text-align: center; font-size: 14px; }
.chunk-content {
  color: var(--text-secondary); font-size: 13px; line-height: 1.7;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-height: 24px;
  transition: all 0.3s ease;
}
.chunk-content.expanded {
  white-space: pre-wrap; max-height: none; overflow: visible;
  text-overflow: unset; background: rgba(59,130,246,0.04);
  padding: 8px 10px; border-radius: 6px; margin-top: 4px;
}

.eval-empty { text-align: center; padding: 60px 20px; color: var(--text-muted); }

/* 响应式 */
@media (max-width: 768px) {
  .eval-header { flex-direction: column; align-items: flex-start; }
  .metrics-grid { grid-template-columns: 1fr 1fr; }
  .result-header { flex-direction: column; align-items: flex-start; }
  .result-metrics { flex-wrap: wrap; }
}
</style>
