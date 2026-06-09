"""spaCy 抽取器：基于 spaCy 原生规则和模板抽取实体与关系

使用 spaCy 框架内的三种规则方式：
1. EntityRuler - 基于模式的实体识别（精确匹配 + Token 属性模式）
2. Matcher - 基于 Token 属性的模式匹配（替代纯正则，支持词性/形态等特征）
3. DependencyMatcher - 基于依存句法树的模式匹配（捕获主谓宾等句法结构）
"""
import logging
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# spaCy 模型和 nlp 实例（延迟加载）
_nlp = None

# ========== EntityRuler 模式 ==========
# 精确字符串匹配 + Token 属性模式
OPS_ENTITY_PATTERNS = [
    # --- 组件：精确匹配 ---
    {"label": "Component", "pattern": "Redis"},
    {"label": "Component", "pattern": "MySQL"},
    {"label": "Component", "pattern": "Nginx"},
    {"label": "Component", "pattern": "MongoDB"},
    {"label": "Component", "pattern": "Elasticsearch"},
    {"label": "Component", "pattern": "Kubernetes"},
    {"label": "Component", "pattern": "K8s"},
    {"label": "Component", "pattern": "Docker"},
    {"label": "Component", "pattern": "Linux"},
    {"label": "Component", "pattern": "Kafka"},
    {"label": "Component", "pattern": "RabbitMQ"},
    {"label": "Component", "pattern": "PostgreSQL"},
    {"label": "Component", "pattern": "Prometheus"},
    {"label": "Component", "pattern": "Grafana"},
    {"label": "Component", "pattern": "Tomcat"},
    {"label": "Component", "pattern": "Apache"},
    {"label": "Component", "pattern": "etcd"},
    {"label": "Component", "pattern": "Zookeeper"},
    {"label": "Component", "pattern": "Hadoop"},
    {"label": "Component", "pattern": "Spark"},
    {"label": "Component", "pattern": "Flink"},
    {"label": "Component", "pattern": "Jenkins"},
    {"label": "Component", "pattern": "GitLab"},
    {"label": "Component", "pattern": "Neo4j"},
    {"label": "Component", "pattern": "Consul"},
    {"label": "Component", "pattern": "Memcached"},
    # --- 组件：Token 属性模式（匹配 "XX服务" "XX集群" 等） ---
    {"label": "Component", "pattern": [
        {"TEXT": {"REGEX": "^(Redis|MySQL|Nginx|MongoDB|Kafka|Redis|PostgreSQL)"}},
        {"TEXT": {"IN": ["服务", "集群", "节点", "实例", "主节点", "从节点"]}},
    ]},
    # --- 故障：精确匹配 ---
    {"label": "Fault", "pattern": "OOM"},
    {"label": "Fault", "pattern": "内存溢出"},
    {"label": "Fault", "pattern": "内存泄漏"},
    {"label": "Fault", "pattern": "内存不足"},
    {"label": "Fault", "pattern": "连接超时"},
    {"label": "Fault", "pattern": "连接失败"},
    {"label": "Fault", "pattern": "CPU过高"},
    {"label": "Fault", "pattern": "CPU满载"},
    {"label": "Fault", "pattern": "磁盘满"},
    {"label": "Fault", "pattern": "磁盘不足"},
    {"label": "Fault", "pattern": "服务不可用"},
    {"label": "Fault", "pattern": "服务宕机"},
    {"label": "Fault", "pattern": "慢查询"},
    {"label": "Fault", "pattern": "主从同步失败"},
    {"label": "Fault", "pattern": "脑裂"},
    # --- 故障：Token 属性模式（匹配 "XX失败/错误/异常"） ---
    {"label": "Fault", "pattern": [
        {"TEXT": {"IN": ["连接", "写入", "读取", "同步", "启动", "认证", "保存", "重写"]}},
        {"TEXT": {"IN": ["失败", "错误", "异常", "拒绝", "中断"]}},
    ]},
    # --- 命令：精确匹配 ---
    {"label": "Command", "pattern": "redis-cli"},
    {"label": "Command", "pattern": "systemctl"},
    {"label": "Command", "pattern": "kubectl"},
    {"label": "Command", "pattern": "docker"},
    {"label": "Command", "pattern": "nginx"},
    # --- 命令：Token 属性模式（匹配 "redis-cli info" 等） ---
    {"label": "Command", "pattern": [
        {"TEXT": "redis-cli"},
        {"TEXT": {"REGEX": "^(info|monitor|slowlog|client|config|ping|cluster)$"}},
    ]},
    {"label": "Command", "pattern": [
        {"TEXT": "systemctl"},
        {"TEXT": {"IN": ["start", "stop", "restart", "status"]}},
        {"OP": "?", "TEXT": {"NOT_IN": ["。", "，", "\n"]}},
    ]},
    {"label": "Command", "pattern": [
        {"TEXT": "kubectl"},
        {"TEXT": {"IN": ["get", "describe", "logs", "exec", "apply", "delete"]}},
    ]},
    # --- 配置项：精确匹配 ---
    {"label": "Config", "pattern": "maxmemory"},
    {"label": "Config", "pattern": "maxclients"},
    {"label": "Config", "pattern": "max_connections"},
    {"label": "Config", "pattern": "innodb_buffer_pool_size"},
    {"label": "Config", "pattern": "worker_connections"},
    {"label": "Config", "pattern": "keepalive_timeout"},
    {"label": "Config", "pattern": "wait_timeout"},
    {"label": "Config", "pattern": "slow_query_log"},
    {"label": "Config", "pattern": "bind-address"},
    {"label": "Config", "pattern": "vm.overcommit_memory"},
    {"label": "Config", "pattern": "tcp-keepalive"},
    {"label": "Config", "pattern": "activedefrag"},
    {"label": "Config", "pattern": "timeout"},
    # --- 配置项：Token 属性模式（匹配 "XX参数/配置项"） ---
    {"label": "Config", "pattern": [
        {"TEXT": {"NOT_IN": ["的", "了", "和", "与", "在", "是"]}},
        {"TEXT": {"IN": ["参数", "配置项", "选项", "阈值"]}},
    ]},
    # --- 监控指标 ---
    {"label": "Metric", "pattern": "CPU使用率"},
    {"label": "Metric", "pattern": "内存使用率"},
    {"label": "Metric", "pattern": "QPS"},
    {"label": "Metric", "pattern": "TPS"},
    {"label": "Metric", "pattern": "命中率"},
    # --- 监控指标：Token 属性模式 ---
    {"label": "Metric", "pattern": [
        {"TEXT": {"IN": ["连接", "延迟", "吞吐", "请求"]}},
        {"TEXT": {"IN": ["数", "量", "率", "时间"]}},
    ]},
]

# ========== Matcher 模式 ==========
# 基于 Token 属性的关系触发词模式匹配
# 每个模式定义：匹配到后，如何从匹配的 span 中提取左右实体

# 因果关系 Matcher 模式
CAUSAL_MATCHER_PATTERNS = [
    # "A 导致 B" / "A 引起 B" / "A 造成 B" - 宽松模式，匹配触发词
    {"label": "causes_trigger", "pattern": [
        {"TEXT": {"IN": ["导致", "引起", "造成", "引发"]}},
    ]},
    # "由于 A ，导致 B"
    {"label": "causes_yuwei", "pattern": [
        {"TEXT": {"IN": ["由于", "因为", "因"]}},
    ]},
    # "A 使得 B" / "A 致使 B"
    {"label": "causes_shide", "pattern": [
        {"TEXT": {"IN": ["使得", "致使", "以至于"]}},
    ]},
]

# 修复关系 Matcher 模式
FIX_MATCHER_PATTERNS = [
    # "通过 A 修复 B" / "使用 A 解决 B"
    {"label": "fixes_trigger", "pattern": [
        {"TEXT": {"IN": ["修复", "解决", "消除", "排除", "恢复"]}},
    ]},
]

# 配置关系 Matcher 模式
CONFIG_MATCHER_PATTERNS = [
    # "A 配置 B" / "A 设置 B" / "A 限制 B"
    {"label": "configures_trigger", "pattern": [
        {"TEXT": {"IN": ["配置", "设置", "控制", "限制", "指定", "调整"]}},
    ]},
    # "修改 A"
    {"label": "configures_modify", "pattern": [
        {"TEXT": {"IN": ["修改", "调整"]}},
    ]},
]

# 检查关系 Matcher 模式
CHECK_MATCHER_PATTERNS = [
    # "使用 A 查看 B" / "用 A 检查 B"
    {"label": "checks_trigger", "pattern": [
        {"TEXT": {"IN": ["检查", "查看", "验证", "确认", "测试", "排查"]}},
    ]},
]

# 监控关系 Matcher 模式
MONITOR_MATCHER_PATTERNS = [
    # "使用 A 监控 B"
    {"label": "monitors_trigger", "pattern": [
        {"TEXT": {"IN": ["监控", "监测", "监视", "观察"]}},
    ]},
]

# ========== DependencyMatcher 模式 ==========
# 基于依存句法树的模式，捕获主谓宾结构

# 因果关系：主语 -(nsubj)-> 谓语(导致) <-dobj-(宾语)
CAUSAL_DEP_PATTERNS = [
    {
        "label": "causes_svo",
        "pattern": [
            # 谓语动词（导致/引起/造成）
            {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"TEXT": {"IN": ["导致", "引起", "造成", "引发"]}}},
            # 主语（nsubj 依存于谓语）
            {"LEFT_ID": "verb", "REL_OP": "<", "RIGHT_ID": "subject", "RIGHT_ATTRS": {"DEP": {"IN": ["nsubj", "nsubjpass"]}}},
            # 宾语（dobj 依存于谓语）
            {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "object", "RIGHT_ATTRS": {"DEP": {"IN": ["dobj", "attr"]}}},
        ],
    },
]

# 修复关系：谓语(修复/解决) <-nsubj-(方式) >dobj>(问题)
FIX_DEP_PATTERNS = [
    {
        "label": "fixes_svo",
        "pattern": [
            {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"TEXT": {"IN": ["修复", "解决", "消除", "排除"]}}},
            {"LEFT_ID": "verb", "REL_OP": "<", "RIGHT_ID": "subject", "RIGHT_ATTRS": {"DEP": {"IN": ["nsubj", "nsubjpass"]}}},
            {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "object", "RIGHT_ATTRS": {"DEP": {"IN": ["dobj", "attr"]}}},
        ],
    },
]

# 配置关系：谓语(配置/设置) <-nsubj-(配置项) >dobj>(目标)
CONFIG_DEP_PATTERNS = [
    {
        "label": "configures_svo",
        "pattern": [
            {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"TEXT": {"IN": ["配置", "设置", "控制", "限制", "调整"]}}},
            {"LEFT_ID": "verb", "REL_OP": "<", "RIGHT_ID": "subject", "RIGHT_ATTRS": {"DEP": {"IN": ["nsubj", "nsubjpass"]}}},
            {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "object", "RIGHT_ATTRS": {"DEP": {"IN": ["dobj", "attr"]}}},
        ],
    },
]


def _get_nlp():
    """延迟加载 spaCy 模型，配置 EntityRuler + Matcher + DependencyMatcher"""
    global _nlp
    if _nlp is not None:
        return _nlp

    try:
        import spacy
        from spacy.matcher import Matcher, DependencyMatcher

        # 尝试加载中文模型
        try:
            nlp = spacy.load("zh_core_web_sm")
        except OSError:
            logger.info("[spaCy] 正在下载 zh_core_web_sm 模型...")
            spacy.cli.download("zh_core_web_sm")  # type: ignore
            nlp = spacy.load("zh_core_web_sm")

        # 1. 添加自定义 EntityRuler（在 ner 之前）
        if "entity_ruler" not in nlp.pipe_names:
            ruler = nlp.add_pipe("entity_ruler", before="ner")
            ruler.add_patterns(OPS_ENTITY_PATTERNS)

        # 2. 添加 Matcher（基于 Token 属性的模式匹配）
        matcher = Matcher(nlp.vocab)
        for pattern_group in [CAUSAL_MATCHER_PATTERNS, FIX_MATCHER_PATTERNS,
                              CONFIG_MATCHER_PATTERNS, CHECK_MATCHER_PATTERNS,
                              MONITOR_MATCHER_PATTERNS]:
            for p in pattern_group:
                matcher.add(p["label"], [p["pattern"]])

        # 3. 添加 DependencyMatcher（基于依存句法的模式匹配）
        dep_matcher = DependencyMatcher(nlp.vocab)
        for pattern_group in [CAUSAL_DEP_PATTERNS, FIX_DEP_PATTERNS, CONFIG_DEP_PATTERNS]:
            for p in pattern_group:
                dep_matcher.add(p["label"], [p["pattern"]])

        _nlp = nlp
        # 将 matcher 和 dep_matcher 存到 nlp 对象上，方便后续使用
        _nlp._ops_matcher = matcher  # type: ignore
        _nlp._ops_dep_matcher = dep_matcher  # type: ignore
        logger.info("[spaCy] 模型加载完成（EntityRuler + Matcher + DependencyMatcher）")
        return _nlp

    except ImportError:
        logger.warning("[spaCy] spacy 未安装，请运行 pip install spacy")
        return None
    except Exception as e:
        logger.warning(f"[spaCy] 模型加载失败: {e}")
        return None


# ========== 关系方向规则 ==========
# 基于实体类型的方向约束，与 spaCy Token 属性结合

def _determine_direction(e1_text: str, e1_type: str, e2_text: str, e2_type: str,
                         relation: str) -> Optional[Tuple[str, str, str, str]]:
    """根据实体类型和关系类型确定方向，返回 None 表示不合理"""
    if relation == "causes":
        if e1_type == "Component" and e2_type == "Fault":
            return e1_text, e1_type, e2_text, e2_type
        elif e1_type == "Fault" and e2_type == "Component":
            return e2_text, e2_type, e1_text, e1_type
        elif e1_type in ("Component", "Fault") and e2_type in ("Component", "Fault"):
            return e1_text, e1_type, e2_text, e2_type
        return None

    elif relation == "fixes":
        if e1_type in ("Command", "Config") and e2_type == "Fault":
            return e1_text, e1_type, e2_text, e2_type
        elif e1_type == "Fault" and e2_type in ("Command", "Config"):
            return e2_text, e2_type, e1_text, e1_type
        return None

    elif relation == "configures":
        if e1_type == "Config" and e2_type in ("Component", "Fault", "Metric"):
            return e1_text, e1_type, e2_text, e2_type
        elif e1_type in ("Component", "Fault", "Metric") and e2_type == "Config":
            return e2_text, e2_type, e1_text, e1_type
        return None

    elif relation == "checks":
        if e1_type == "Command" and e2_type in ("Component", "Fault", "Config"):
            return e1_text, e1_type, e2_text, e2_type
        elif e1_type in ("Component", "Fault", "Config") and e2_type == "Command":
            return e2_text, e2_type, e1_text, e1_type
        return None

    elif relation == "monitors":
        if e1_type in ("Command", "Component") and e2_type in ("Component", "Metric"):
            return e1_text, e1_type, e2_text, e2_type
        elif e1_type in ("Component", "Metric") and e2_type in ("Command", "Component"):
            return e2_text, e2_type, e1_text, e1_type
        return None

    elif relation == "indicates":
        if e1_type == "Metric" and e2_type == "Fault":
            return e1_text, e1_type, e2_text, e2_type
        elif e1_type == "Fault" and e2_type == "Metric":
            return e2_text, e2_type, e1_text, e1_type
        return None

    elif relation == "depends_on":
        if e1_type == "Component" and e2_type == "Component":
            return e1_text, e1_type, e2_text, e2_type
        return None

    return None


def _find_entity_in_span(span, ent_type: Optional[str] = None) -> Optional[Dict]:
    """在 span 中找到第一个指定类型的实体"""
    for ent in span.ents:
        if ent_type and ent.label_ != ent_type:
            continue
        return {"name": ent.text.strip(), "type": ent.label_}
    return None


def _find_entities_in_span(span) -> List[Dict]:
    """在 span 中找到所有运维领域实体"""
    accepted = {"Component", "Fault", "Command", "Config", "Metric"}
    entities = []
    seen = set()
    for ent in span.ents:
        if ent.label_ in accepted and len(ent.text.strip()) <= 30:
            name = ent.text.strip()
            key = f"{name}:{ent.label_}"
            if key not in seen and name:
                seen.add(key)
                entities.append({"name": name, "type": ent.label_})
    return entities


# ========== 抽取函数 ==========

def extract_entities(text: str) -> List[Dict]:
    """使用 spaCy EntityRuler 从文本中抽取实体

    EntityRuler 支持精确匹配和 Token 属性模式两种方式。
    """
    nlp = _get_nlp()
    if nlp is None:
        return []

    doc = nlp(text)
    accepted_labels = {"Component", "Fault", "Command", "Config", "Metric"}
    entities = []
    seen = set()

    for ent in doc.ents:
        label = ent.label_
        text_val = ent.text.strip()
        if not text_val or label not in accepted_labels or len(text_val) > 30:
            continue
        key = f"{text_val}:{label}"
        if key not in seen:
            seen.add(key)
            entities.append({"name": text_val, "type": label})

    return entities


def _extract_with_matcher(doc) -> List[Dict]:
    """方式1: 使用 spaCy Matcher 抽取关系

    Matcher 匹配到触发词后，在触发词所在的句子中，
    利用 spaCy 的实体标注结果，找触发词两侧最近的实体对。
    """
    nlp = _get_nlp()
    matcher = getattr(nlp, '_ops_matcher', None)
    if matcher is None:
        return []

    triples = []
    seen = set()

    # Matcher 匹配结果中的 label -> relation 映射
    label_relation_map = {
        "causes_trigger": "causes", "causes_yuwei": "causes", "causes_shide": "causes",
        "fixes_trigger": "fixes",
        "configures_trigger": "configures", "configures_modify": "configures",
        "checks_trigger": "checks",
        "monitors_trigger": "monitors",
    }

    matches = matcher(doc)
    for match_id, start, end in matches:
        label = nlp.vocab.strings[match_id]
        relation = label_relation_map.get(label)
        if not relation:
            continue

        # 触发词 token
        trigger_token = doc[start]
        trigger_idx = trigger_token.i

        # 在触发词所在句子中找实体
        sent = trigger_token.sent
        sent_entities = []
        for ent in doc.ents:
            if ent.label_ in {"Component", "Fault", "Command", "Config", "Metric"} \
               and ent.start >= sent.start and ent.end <= sent.end \
               and len(ent.text.strip()) <= 30:
                # 用实体 root token 的位置来判断左右
                ent_center = ent.root.i
                sent_entities.append((ent_center, ent.text.strip(), ent.label_))

        if len(sent_entities) < 2:
            continue

        # 按位置排序
        sent_entities.sort(key=lambda x: x[0])

        # 找触发词左侧和右侧最近的实体
        left_entities = [(pos, name, typ) for pos, name, typ in sent_entities if pos < trigger_idx]
        right_entities = [(pos, name, typ) for pos, name, typ in sent_entities if pos >= trigger_idx]

        if not left_entities or not right_entities:
            continue

        # 优先找能构成有效关系的实体对
        # 根据右侧实体类型动态调整关系（如"配置"+Fault=fixes）
        verb_type_override = {
            ("配置", "Fault"): "fixes", ("设置", "Fault"): "fixes",
            ("控制", "Fault"): "fixes", ("限制", "Fault"): "fixes",
            ("修改", "Fault"): "fixes", ("调整", "Fault"): "fixes",
        }
        # 遍历所有左右实体对，找最合理的（优先近的，但允许跳过不合理的）
        found_pairs = []
        for _, e1_name, e1_type in reversed(left_entities):  # 左侧从近到远
            for _, e2_name, e2_type in right_entities:  # 右侧从近到远
                if e1_name == e2_name:
                    continue
                actual_relation = verb_type_override.get((trigger_token.text, e2_type), relation)
                result = _determine_direction(e1_name, e1_type, e2_name, e2_type, actual_relation)
                if result is not None:
                    found_pairs.append((actual_relation, result))

        # 按优先级选：fixes > causes > configures > checks > monitors
        priority = {"fixes": 0, "causes": 1, "configures": 2, "checks": 3, "monitors": 4, "depends_on": 5, "indicates": 6}
        found_pairs.sort(key=lambda x: priority.get(x[0], 99))
        for actual_relation, (from_e, from_t, to_e, to_t) in found_pairs[:2]:  # 最多取2个
            key = f"{from_e}|{actual_relation}|{to_e}"
            if key not in seen:
                seen.add(key)
                triples.append({
                    "from_entity": from_e, "from_type": from_t,
                    "relation": actual_relation,
                    "to_entity": to_e, "to_type": to_t,
                })

    return triples


def _extract_with_dep_matcher(doc) -> List[Dict]:
    """方式2: 使用 spaCy DependencyMatcher 抽取关系

    DependencyMatcher 基于依存句法树进行模式匹配，
    能捕获主谓宾等深层句法结构，比表面模式更准确。
    """
    nlp = _get_nlp()
    dep_matcher = getattr(nlp, '_ops_dep_matcher', None)
    if dep_matcher is None:
        return []

    triples = []
    seen = set()

    # dep matcher label -> relation 映射
    label_relation_map = {
        "causes_svo": "causes",
        "fixes_svo": "fixes",
        "configures_svo": "configures",
    }

    matches = dep_matcher(doc)
    for match_id, token_ids in matches:
        label = nlp.vocab.strings[match_id]
        relation = label_relation_map.get(label)
        if not relation:
            continue

        # token_ids: [verb_idx, subject_idx, object_idx]
        if len(token_ids) < 3:
            continue

        verb_token = doc[token_ids[0]]
        subj_token = doc[token_ids[1]]
        obj_token = doc[token_ids[2]]

        # 在主语和宾语的子树中查找实体
        subj_entity = _find_entity_in_subtree(subj_token)
        obj_entity = _find_entity_in_subtree(obj_token)

        if not subj_entity or not obj_entity:
            continue
        if subj_entity["name"] == obj_entity["name"]:
            continue

        result = _determine_direction(
            subj_entity["name"], subj_entity["type"],
            obj_entity["name"], obj_entity["type"],
            relation
        )
        if result is None:
            continue

        from_e, from_t, to_e, to_t = result
        key = f"{from_e}|{relation}|{to_e}"
        if key not in seen:
            seen.add(key)
            triples.append({
                "from_entity": from_e, "from_type": from_t,
                "relation": relation,
                "to_entity": to_e, "to_type": to_t,
            })

    return triples


def _find_entity_in_subtree(token) -> Optional[Dict]:
    """在 token 及其子树中查找运维领域实体"""
    accepted = {"Component", "Fault", "Command", "Config", "Metric"}
    # 先检查 token 自身
    if token.ent_type_ in accepted:
        return {"name": token.text.strip(), "type": token.ent_type_}
    # 再检查子树
    for child in token.subtree:
        if child.ent_type_ in accepted:
            return {"name": child.text.strip(), "type": child.ent_type_}
    return None


def _extract_with_token_rules(doc) -> List[Dict]:
    """方式3: 基于 Token 依存关系和词性的规则抽取

    利用 spaCy 的 Token 属性（dep_, pos_, head 等），
    通过规则判断实体间的关系，无需正则。

    核心思路：
    1. 收集所有实体及其 root token 的 head 信息
    2. 通过 head 的文本判断关系类型
    3. 通过实体在句子中的位置确定方向
    """
    triples = []
    seen = set()
    accepted = {"Component", "Fault", "Command", "Config", "Metric"}

    # 收集所有实体信息
    entity_info = []
    for ent in doc.ents:
        if ent.label_ not in accepted or len(ent.text.strip()) > 30:
            continue
        entity_info.append({
            "name": ent.text.strip(),
            "type": ent.label_,
            "root": ent.root,          # 实体的 root token
            "start": ent.start,        # 实体在 doc 中的起始位置
            "end": ent.end,            # 实体在 doc 中的结束位置
        })

    if len(entity_info) < 2:
        return []

    # 触发词 -> 关系类型映射（基础映射）
    verb_relations = {
        "导致": "causes", "引起": "causes", "造成": "causes", "引发": "causes",
        "修复": "fixes", "解决": "fixes", "消除": "fixes", "排除": "fixes",
        "配置": "configures", "设置": "configures", "控制": "configures", "限制": "configures",
        "检查": "checks", "查看": "checks", "验证": "checks", "确认": "checks",
        "监控": "monitors", "监测": "monitors", "监视": "monitors",
        "依赖": "depends_on", "需要": "depends_on",
        "修改": "configures", "调整": "configures",
    }

    # 触发词 + 右侧实体类型 -> 关系类型覆盖映射
    # 例如："配置" + Fault = fixes（配置项修复故障），而非 configures
    verb_type_override = {
        ("配置", "Fault"): "fixes",
        ("设置", "Fault"): "fixes",
        ("控制", "Fault"): "fixes",
        ("限制", "Fault"): "fixes",
        ("修改", "Fault"): "fixes",
        ("调整", "Fault"): "fixes",
    }

    # 规则1: 遍历所有触发词 token，找其左右两侧的实体
    for token in doc:
        base_relation = verb_relations.get(token.text)
        if not base_relation:
            continue

        # 找触发词左侧和右侧的实体
        left_entities = []
        right_entities = []
        for e in entity_info:
            if e["end"] <= token.i:
                left_entities.append(e)
            elif e["start"] > token.i:
                right_entities.append(e)

        if not left_entities or not right_entities:
            continue

        # 取触发词两侧的实体，优先选能构成有效关系的实体对
        # 不只取最近的，而是遍历左侧和右侧所有实体找最合理的配对
        found_pairs = []
        for left_e in reversed(left_entities):
            for right_e in right_entities:
                if left_e["name"] == right_e["name"]:
                    continue
                # 根据右侧实体类型动态调整关系
                relation = verb_type_override.get((token.text, right_e["type"]), base_relation)
                result = _determine_direction(left_e["name"], left_e["type"],
                                             right_e["name"], right_e["type"], relation)
                if result is not None:
                    found_pairs.append((relation, result))

        # 按优先级选：fixes > causes > configures > checks > monitors
        priority = {"fixes": 0, "causes": 1, "configures": 2, "checks": 3, "monitors": 4, "depends_on": 5, "indicates": 6}
        found_pairs.sort(key=lambda x: priority.get(x[0], 99))
        for relation, (from_e, from_t, to_e, to_t) in found_pairs[:2]:  # 最多取2个
            key = f"{from_e}|{relation}|{to_e}"
            if key not in seen:
                seen.add(key)
                triples.append({
                    "from_entity": from_e, "from_type": from_t,
                    "relation": relation,
                    "to_entity": to_e, "to_type": to_t,
                })

    # 规则2: 如果两个实体的 root token 共享同一个 head verb
    for i, e1 in enumerate(entity_info):
        for e2 in entity_info[i + 1:]:
            # 检查 e1.root.head 和 e2.root.head 是否相同
            head1 = e1["root"].head
            head2 = e2["root"].head

            # 同一个 head
            if head1.i == head2.i:
                base_relation = verb_relations.get(head1.text)
                if base_relation:
                    # 根据实体类型动态调整关系
                    relation = verb_type_override.get((head1.text, e2["type"]), base_relation)
                    result = _determine_direction(e1["name"], e1["type"],
                                                 e2["name"], e2["type"], relation)
                    if result:
                        from_e, from_t, to_e, to_t = result
                        key = f"{from_e}|{relation}|{to_e}"
                        if key not in seen:
                            seen.add(key)
                            triples.append({
                                "from_entity": from_e, "from_type": from_t,
                                "relation": relation,
                                "to_entity": to_e, "to_type": to_t,
                            })

            # e1 的 head 是 e2 的 head 的 head（间接依存）
            elif head1.i == head2.head.i:
                base_relation = verb_relations.get(head2.head.text)
                if base_relation:
                    relation = verb_type_override.get((head2.head.text, e2["type"]), base_relation)
                    result = _determine_direction(e1["name"], e1["type"],
                                                 e2["name"], e2["type"], relation)
                    if result:
                        from_e, from_t, to_e, to_t = result
                        key = f"{from_e}|{relation}|{to_e}"
                        if key not in seen:
                            seen.add(key)
                            triples.append({
                                "from_entity": from_e, "from_type": from_t,
                                "relation": relation,
                                "to_entity": to_e, "to_type": to_t,
                            })
            elif head2.i == head1.head.i:
                base_relation = verb_relations.get(head1.head.text)
                if base_relation:
                    relation = verb_type_override.get((head1.head.text, e1["type"]), base_relation)
                    result = _determine_direction(e1["name"], e1["type"],
                                                 e2["name"], e2["type"], relation)
                    if result:
                        from_e, from_t, to_e, to_t = result
                        key = f"{from_e}|{relation}|{to_e}"
                        if key not in seen:
                            seen.add(key)
                            triples.append({
                                "from_entity": from_e, "from_type": from_t,
                                "relation": relation,
                                "to_entity": to_e, "to_type": to_t,
                            })

    return triples


def extract_relations(text: str) -> List[Dict]:
    """使用 spaCy 三种规则方式抽取关系

    1. Matcher: 基于 Token 属性的模式匹配
    2. DependencyMatcher: 基于依存句法树的模式匹配
    3. Token 规则: 基于依存关系和词性的规则判断
    """
    nlp = _get_nlp()
    if nlp is None:
        return []

    doc = nlp(text)
    all_triples = []
    seen = set()

    def _add(triples: List[Dict]):
        for t in triples:
            key = f"{t['from_entity']}|{t['relation']}|{t['to_entity']}"
            if key not in seen:
                seen.add(key)
                all_triples.append(t)

    # 方式1: Matcher 模式匹配
    matcher_triples = _extract_with_matcher(doc)
    _add(matcher_triples)
    logger.debug(f"[spaCy Matcher] 抽取 {len(matcher_triples)} 个")

    # 方式2: DependencyMatcher 依存句法匹配
    dep_triples = _extract_with_dep_matcher(doc)
    _add(dep_triples)
    logger.debug(f"[spaCy DepMatcher] 抽取 {len(dep_triples)} 个")

    # 方式3: Token 依存规则
    token_triples = _extract_with_token_rules(doc)
    _add(token_triples)
    logger.debug(f"[spaCy Token规则] 抽取 {len(token_triples)} 个")

    return all_triples


def extract_triples(text: str) -> List[Dict]:
    """使用 spaCy 从文本中抽取所有三元组

    Returns:
        三元组列表，每个元素包含 from_entity, from_type, relation, to_entity, to_type
    """
    nlp = _get_nlp()
    if nlp is None:
        logger.warning("[spaCy] NLP 模型不可用，跳过抽取")
        return []

    triples = extract_relations(text)

    if triples:
        logger.info(f"[spaCy抽取] 从文本中抽取 {len(triples)} 个三元组")
    return triples
