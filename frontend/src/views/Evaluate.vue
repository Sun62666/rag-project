<template>
  <div class="evaluate-view">
    <div class="eval-header">
      <div>
        <h2 style="margin:0;color:var(--text-primary)">检索效果评估</h2>
        <span style="color:var(--text-muted);font-size:13px">Precision@10 指标评估，基于余弦相似度</span>
      </div>
      <div class="eval-controls">
        <span style="font-size:13px;color:var(--text-muted)">相似度阈值:</span>
        <input type="number" v-model.number="threshold" min="0" max="1" step="0.05" class="threshold-input" />
        <button class="eval-run-btn" :disabled="evaluating" @click="runEvaluation">
          <i class="fa-solid fa-chart-line"></i> {{ evaluating ? '评估中...' : '运行评估' }}
        </button>
      </div>
    </div>

    <div class="eval-questions" v-if="evalQuestions.length > 0">
      <h4 style="color:var(--text-primary);margin-bottom:10px">预定义评估问题</h4>
      <div v-for="(eq, i) in evalQuestions" :key="i" class="eq-row">
        <strong style="color:var(--accent)">Q{{ i + 1 }}:</strong> {{ eq.question }}
      </div>
    </div>

    <div v-if="evalResult" class="eval-results">
      <div :class="['eval-summary', evalResult.avg_precision_at_10 >= 0.7 ? 'good' : evalResult.avg_precision_at_10 >= 0.4 ? 'warn' : 'bad']">
        <div class="summary-value">平均 Precision@10: {{ evalResult.avg_precision_at_10 }}</div>
        <div class="summary-detail">阈值: {{ evalResult.threshold }} | 共 {{ evalResult.results.length }} 个问题</div>
      </div>

      <div v-for="(result, i) in evalResult.results" :key="i" class="result-card">
        <div class="result-header" @click="toggleExpand(i)" style="cursor:pointer">
          <div style="display:flex;align-items:center;gap:8px">
            <i :class="expandedCards[i] ? 'fa-solid fa-chevron-down' : 'fa-solid fa-chevron-right'" style="color:var(--text-dim);font-size:12px;width:16px"></i>
            <strong style="color:var(--text-primary)">Q{{ i + 1 }}: {{ result.question }}</strong>
          </div>
          <span :class="['precision-tag', result.precision_at_10 >= 0.7 ? 'good' : result.precision_at_10 >= 0.4 ? 'warn' : 'bad']">
            Precision@10: {{ result.precision_at_10 }} (TP: {{ result.tp_count }}/10)
          </span>
        </div>
        <div v-show="expandedCards[i]" class="chunks-list">
          <div v-for="(chunk, ci) in result.chunks" :key="ci" class="chunk-row" @click="toggleChunk(i, ci)" style="cursor:pointer">
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

    <div v-else class="eval-empty">
      <i class="fa-solid fa-chart-line" style="font-size:48px;color:var(--text-dim);margin-bottom:12px"></i>
      <div>点击"运行评估"按钮开始检索效果评估</div>
      <div style="font-size:13px;color:var(--text-dim);margin-top:4px">请先上传文档到知识库</div>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted } from 'vue'
import { evaluateAPI } from '../api'
import { ElMessage } from 'element-plus'

defineOptions({ name: 'Evaluate' })

const threshold = ref(0.7)
const evaluating = ref(false)
const evalResult = ref(null)
const evalQuestions = ref([])
const expandedCards = reactive({})
const expandedChunks = reactive({})

const toggleExpand = (i) => {
  expandedCards[i] = !expandedCards[i]
}

const toggleChunk = (i, ci) => {
  const key = `${i}-${ci}`
  expandedChunks[key] = !expandedChunks[key]
}

const loadQuestions = async () => {
  try {
    const res = await evaluateAPI.getQuestions()
    evalQuestions.value = res.data.questions || []
  } catch (e) { console.error('加载评估问题失败:', e) }
}

const runEvaluation = async () => {
  evaluating.value = true
  try {
    const res = await evaluateAPI.run(threshold.value)
    if (res.data.status === 'ok') {
      evalResult.value = res.data
      // 默认展开所有问题卡片
      if (res.data.results) {
        res.data.results.forEach((_, i) => { expandedCards[i] = true })
      }
      ElMessage.success('评估完成')
    } else {
      ElMessage.error(res.data.message || '评估失败')
    }
  } catch (e) {
    ElMessage.error('评估请求失败: ' + (e.response?.data?.detail || e.message))
  } finally { evaluating.value = false }
}

onMounted(() => { loadQuestions() })
</script>

<style scoped>
.evaluate-view { padding: 24px 28px; display: flex; flex-direction: column; gap: 20px; }

.eval-header { display: flex; justify-content: space-between; align-items: center; }
.eval-controls { display: flex; align-items: center; gap: 8px; }
.threshold-input {
  width: 80px; background: var(--bg-primary); border: 1px solid var(--border);
  border-radius: 6px; padding: 6px 10px; color: var(--text-primary); font-size: 13px; outline: none;
}
.threshold-input:focus { border-color: var(--accent); }
.eval-run-btn {
  padding: 8px 16px; border-radius: 8px; border: none;
  background: linear-gradient(135deg, #3B82F6, #6366F1); color: #fff;
  font-size: 13px; cursor: pointer; transition: all 0.2s;
  display: flex; align-items: center; gap: 6px;
}
.eval-run-btn:hover:not(:disabled) { transform: translateY(-1px); box-shadow: 0 4px 16px rgba(59,130,246,0.35); }
.eval-run-btn:disabled { opacity: 0.5; cursor: not-allowed; }

.eval-questions { background: var(--bg-secondary); border: 1px solid var(--border); border-radius: 12px; padding: 16px 20px; }
.eq-row { padding: 8px 12px; background: var(--bg-primary); border-radius: 6px; margin-bottom: 6px; font-size: 13px; color: var(--text-secondary); }

.eval-results { display: flex; flex-direction: column; gap: 16px; }
.eval-summary { padding: 16px 20px; border-radius: 10px; border-left: 4px solid; }
.eval-summary.good { background: rgba(34,197,94,0.08); border-color: #22C55E; }
.eval-summary.warn { background: rgba(234,179,8,0.08); border-color: #EAB308; }
.eval-summary.bad { background: rgba(239,68,68,0.08); border-color: #EF4444; }
.summary-value { font-size: 18px; font-weight: 700; color: var(--text-primary); }
.summary-detail { font-size: 13px; color: var(--text-muted); margin-top: 4px; }

.result-card { background: var(--bg-secondary); border: 1px solid var(--border); border-left: 3px solid var(--accent); border-radius: 10px; padding: 16px 20px; }
.result-header { display: flex; justify-content: space-between; align-items: center; }
.precision-tag { padding: 4px 10px; border-radius: 4px; font-size: 12px; font-weight: 600; white-space: nowrap; }
.precision-tag.good { background: rgba(34,197,94,0.12); color: #4ade80; }
.precision-tag.warn { background: rgba(234,179,8,0.12); color: #facc15; }
.precision-tag.bad { background: rgba(239,68,68,0.12); color: #f87171; }

.chunks-list { display: flex; flex-direction: column; gap: 6px; margin-top: 12px; }
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
</style>