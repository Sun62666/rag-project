"""规则模板抽取器：基于正则和模式匹配从运维文档中抽取实体和关系

适用于结构化的运维文档，如：
- 故障排查文档（现象→原因→解决方案）
- 配置参考文档（组件→配置项→说明）
- 命令手册（命令→功能→参数）
"""
import re
import logging
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)

# ========== 实体识别模式 ==========

# 组件名称模式
COMPONENT_PATTERNS = {
    "Redis": r'Redis',
    "MySQL": r'MySQL',
    "Nginx": r'Nginx|nginx',
    "MongoDB": r'MongoDB',
    "Elasticsearch": r'Elasticsearch|ElasticSearch',
    "Kubernetes": r'Kubernetes|K8s|k8s',
    "Docker": r'Docker|docker',
    "Linux": r'Linux',
    "CentOS": r'CentOS',
    "Ubuntu": r'Ubuntu',
    "Kafka": r'Kafka',
    "RabbitMQ": r'RabbitMQ',
    "PostgreSQL": r'PostgreSQL|Postgres',
    "Prometheus": r'Prometheus',
    "Grafana": r'Grafana',
    "Tomcat": r'Tomcat',
    "Apache": r'Apache',
    "etcd": r'etcd',
    "Zookeeper": r'Zookeeper',
    "Hadoop": r'\bHadoop\b',
    "Spark": r'\bSpark\b',
    "Flink": r'\bFlink\b',
    "Jenkins": r'\bJenkins\b',
    "GitLab": r'\bGitLab\b',
    "Neo4j": r'\bNeo4j\b',
    "Memcached": r'\bMemcached\b',
    "Consul": r'\bConsul\b',
}

# 故障现象模式
FAULT_PATTERNS = [
    # OOM 类
    (r'OOM', "OOM"),
    (r'OutOfMemory', "OOM"),
    (r'内存溢出', "内存溢出"),
    (r'内存泄漏', "内存泄漏"),
    (r'内存不足', "内存不足"),
    # 连接类
    (r'连接超时', "连接超时"),
    (r'连接失败', "连接失败"),
    (r'连接数满', "连接数满"),
    (r'连接数过多', "连接数过多"),
    (r'Connection\s+refused', "连接被拒绝"),
    (r'Too many connections', "连接数过多"),
    (r'ERR max number of clients reached', "连接数满"),
    # CPU 类
    (r'CPU\s*(使用率)?\s*(过高|满载|100%)', "CPU过高"),
    (r'CPU\s*满', "CPU过高"),
    # 磁盘类
    (r'磁盘(空间)?满', "磁盘满"),
    (r'磁盘(空间)?不足', "磁盘不足"),
    (r'No space left', "磁盘满"),
    # 网络类
    (r'网络(连接)?超时', "网络超时"),
    (r'网络(连接)?中断', "网络中断"),
    # 服务类
    (r'服务不可用', "服务不可用"),
    (r'502错误|502', "502错误"),
    (r'503错误|503', "503错误"),
    (r'500错误|500', "500错误"),
    (r'服务宕机', "服务宕机"),
    (r'服务崩溃', "服务崩溃"),
    # 性能类
    (r'响应延迟(高|过长)', "响应延迟高"),
    (r'性能下降', "性能下降"),
    (r'慢查询', "慢查询"),
    # 持久化类
    (r'RDB\s*(保存|写入)?\s*失败', "RDB保存失败"),
    (r'AOF\s*(重写)?\s*失败', "AOF重写失败"),
    # 复制类
    (r'复制(中断|延迟|失败)', "复制中断"),
    (r'主从同步(失败|中断)', "主从同步失败"),
    # 集群类
    (r'集群(节点)?故障', "集群节点故障"),
    (r'脑裂', "脑裂"),
    # 安全类
    (r'未授权访问', "未授权访问"),
    (r'数据泄露', "数据泄露"),
]

# 命令模式
COMMAND_PATTERNS = [
    # 系统命令
    (r'\b(top|htop|iotop|atop)\b', "Command"),
    (r'\b(free\s+-[hm])\b', "Command"),
    (r'\b(df\s+-[hm])\b', "Command"),
    (r'\b(netstat\s+-\S+)', "Command"),
    (r'\b(ss\s+-\S+)', "Command"),
    (r'\b(ps\s+aux)\b', "Command"),
    (r'\b(systemctl\s+(?:start|stop|restart|status)\s+\S+)', "Command"),
    (r'\b(service\s+\S+\s+(?:start|stop|restart|status))', "Command"),
    # Redis 命令
    (r'(redis-cli\s+\w+)', "Command"),
    (r'\b(INFO|SLOWLOG|BGREWRITEAOF|BGSAVE|FLUSHALL|FLUSHDB|SCAN|KEYS)\b', "Command"),
    # MySQL 命令
    (r'\b(mysql\s+-[uhp]\S+(?:\s+\S+)*)', "Command"),
    (r'\b(SHOW\s+(?:DATABASES|TABLES|GRANTS|PROCESSLIST|STATUS))', "Command"),
    (r'\b(GRANT\s+\S+)', "Command"),
    (r'\b(REVOKE\s+\S+)', "Command"),
    (r'\b(FLUSH\s+PRIVILEGES)', "Command"),
    # Nginx 命令
    (r'\b(nginx\s+-[ts])\b', "Command"),
    (r'\b(nginx\s+-s\s+reload)', "Command"),
    # Docker 命令
    (r'\b(docker\s+(?:ps|logs|exec|run|stop|restart|inspect)\s+\S+)', "Command"),
    # K8s 命令
    (r'\b(kubectl\s+(?:get|describe|logs|exec|apply|delete)\s+\S+)', "Command"),
    # 通用
    (r'\b(ping\s+\S+)', "Command"),
    (r'\b(curl\s+\S+)', "Command"),
    (r'\b(telnet\s+\S+)', "Command"),
    (r'\b(iptables\s+-\S+)', "Command"),
    (r'\b(firewall-cmd\s+\S+)', "Command"),
    (r'\b(chown\s+\S+)', "Command"),
    (r'\b(chmod\s+\S+)', "Command"),
    (r'\b(cat\s+/proc/\S+)', "Command"),
    (r'\b(echo\s+\d+\s*>\s*/proc/\S+)', "Command"),
]

# 配置项模式
CONFIG_PATTERNS = [
    # Redis
    (r'maxmemory(?:-policy)?(?:\s+\S+)?', "Redis"),
    (r'maxclients', "Redis"),
    (r'timeout\s+\d+', "Redis"),
    (r'tcp-keepalive\s+\d+', "Redis"),
    (r'activedefrag\s+\w+', "Redis"),
    (r'auto-aof-rewrite-\w+', "Redis"),
    (r'save\s+\d+\s+\d+', "Redis"),
    (r'bind\s+[\d.]+', "Redis"),
    (r'requirepass\s+\S+', "Redis"),
    # MySQL
    (r'max_connections', "MySQL"),
    (r'innodb_buffer_pool_size', "MySQL"),
    (r'key_buffer_size', "MySQL"),
    (r'thread_cache_size', "MySQL"),
    (r'query_cache_size', "MySQL"),
    (r'wait_timeout', "MySQL"),
    (r'bind-address', "MySQL"),
    (r'slow_query_log', "MySQL"),
    (r'datadir', "MySQL"),
    (r'socket', "MySQL"),
    (r'port\s+\d+', "MySQL"),
    # Nginx
    (r'worker_connections', "Nginx"),
    (r'worker_processes', "Nginx"),
    (r'keepalive_timeout', "Nginx"),
    (r'proxy_read_timeout', "Nginx"),
    (r'proxy_connect_timeout', "Nginx"),
    # Linux 系统
    (r'vm\.overcommit_memory', "Linux"),
    (r'vm\.swappiness', "Linux"),
    (r'net\.core\.somaxconn', "Linux"),
    (r'fs\.file-max', "Linux"),
    (r'ulimit\s+-[na]', "Linux"),
    # K8s
    (r'resource[s]?\.(?:limits|requests)\.(?:cpu|memory)', "Kubernetes"),
    (r'replicas', "Kubernetes"),
    # Docker
    (r'--memory(?:-swap)?', "Docker"),
    (r'--cpus', "Docker"),
]

# ========== 关系抽取模式 ==========

# 因果关系模式
CAUSAL_PATTERNS = [
    (r'(\S+)\s*(?:导致|引起|造成|引发)\s*(\S+)', "causes"),
    (r'(?:由于|因为|因)\s*(\S+?)\s*(?:，|,|导致|引起|造成|使得|引发)\s*(\S+)', "causes"),
    (r'(\S+)\s*(?:使得|致使|以至于)\s*(\S+)', "causes"),
]

# 修复关系模式
FIX_PATTERNS = [
    (r'(?:通过|使用|用|采用)\s*(\S+?)\s*(?:修复|解决|消除|排除)\s*(\S+)', "fixes"),
    (r'(\S+?)\s*(?:可以|可|能)\s*(?:修复|解决|消除|排除)\s*(\S+)', "fixes"),
    (r'(?:修改|调整|设置|配置)\s*(\S+?)\s*(?:可以|可|来|以)\s*(?:修复|解决|避免|防止)\s*(\S+)', "fixes"),
]

# 配置关系模式
CONFIGURE_PATTERNS = [
    (r'(\S+?)\s*(?:配置|设置|控制|限制|指定|定义)\s*(\S+)', "configures"),
    (r'(?:在|通过)\s*(\S+?)\s*(?:中)?配置\s*(\S+)', "configures"),
    (r'(\S+?)\s*(?:参数|选项|配置项)\s*(\S+)', "configures"),
]

# 依赖关系模式
DEPEND_PATTERNS = [
    (r'(\S+?)\s*(?:依赖|依赖于|需要|基于)\s*(\S+)', "depends_on"),
    (r'(\S+?)\s*(?:基于|运行在|部署在)\s*(\S+)', "depends_on"),
]

# 监控关系模式
MONITOR_PATTERNS = [
    (r'(\S+?)\s*(?:监控|监测|监视|观察)\s*(\S+)', "monitors"),
    (r'(?:使用|用)\s*(\S+?)\s*(?:监控|监测|查看|检查)\s*(\S+)', "monitors"),
]

# 检查关系模式
CHECK_PATTERNS = [
    (r'(?:使用|用|执行|运行)\s*(\S+?)\s*(?:检查|查看|验证|确认|测试)\s*(\S+)', "checks"),
    (r'(\S+?)\s*(?:检查|查看|验证|确认)\s*(\S+)', "checks"),
]

# 指示关系模式
INDICATE_PATTERNS = [
    (r'(\S+?)\s*(?:表示|说明|指示|意味着)\s*(\S+)', "indicates"),
    (r'(?:当|如果)\s*(\S+?)\s*(?:出现|达到|超过)\s*.*?(?:说明|表示|意味着|指示)\s*(\S+)', "indicates"),
]


def extract_components(text: str) -> List[Dict]:
    """从文本中抽取组件实体"""
    entities = []
    seen = set()
    for name, pattern in COMPONENT_PATTERNS.items():
        if re.search(pattern, text, re.IGNORECASE):
            if name not in seen:
                seen.add(name)
                entities.append({"name": name, "type": "Component"})
    return entities


def extract_faults(text: str) -> List[Dict]:
    """从文本中抽取故障实体"""
    entities = []
    seen = set()
    for pattern, fault_name in FAULT_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            if fault_name not in seen:
                seen.add(fault_name)
                entities.append({"name": fault_name, "type": "Fault"})
    return entities


def extract_commands(text: str) -> List[Dict]:
    """从文本中抽取命令实体"""
    entities = []
    seen = set()
    for pattern, entity_type in COMMAND_PATTERNS:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for m in matches:
            cmd = m.group(1).strip() if m.lastindex else m.group(0).strip()
            if cmd and cmd not in seen and len(cmd) > 1:
                seen.add(cmd)
                entities.append({"name": cmd, "type": "Command"})
    # 过滤子串重复：如果 "redis-cli info" 已存在，则移除 "info"
    filtered = []
    for e in entities:
        is_substring = False
        for other in entities:
            if e["name"] != other["name"] and e["name"] in other["name"]:
                is_substring = True
                break
        if not is_substring:
            filtered.append(e)
    return filtered


def extract_configs(text: str) -> List[Dict]:
    """从文本中抽取配置项实体"""
    entities = []
    seen = set()
    for pattern, component in CONFIG_PATTERNS:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for m in matches:
            config_name = m.group(0).strip().split()[0]  # 取配置项名称部分
            if config_name and config_name not in seen:
                seen.add(config_name)
                entities.append({"name": config_name, "type": "Config"})
    return entities


def extract_all_entities(text: str) -> List[Dict]:
    """抽取所有实体"""
    entities = []
    seen = set()
    for extractor in [extract_components, extract_faults, extract_commands, extract_configs]:
        for e in extractor(text):
            key = f"{e['name']}:{e['type']}"
            if key not in seen:
                seen.add(key)
                entities.append(e)
    return entities


def _match_relation_patterns(text: str, patterns: List[Tuple], relation_type: str) -> List[Dict]:
    """通用关系模式匹配 - 基于已知实体 + 触发词位置关系"""
    entities = extract_all_entities(text)
    if len(entities) < 2:
        return []

    triples = []
    seen = set()

    # 提取该关系类型对应的所有触发词
    triggers = set()
    for pattern, rel_type in patterns:
        if rel_type == relation_type:
            trigger_match = re.findall(
                r'(导致|引起|造成|引发|使得|致使|以至于|修复|解决|消除|排除|'
                r'配置|设置|控制|限制|指定|定义|依赖|依赖于|需要|基于|'
                r'监控|监测|监视|观察|检查|查看|验证|确认|测试|'
                r'表示|说明|指示|意味着)', pattern)
            triggers.update(trigger_match)

    if not triggers:
        return []

    # 将文本按逗号、句号、换行等分段，更细粒度
    segments = re.split(r'[，,。\n；！!?？]', text)

    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue

        # 找出该段中出现的实体
        seg_entities = [e for e in entities if e["name"] in seg]
        if len(seg_entities) < 2:
            continue

        # 检查该段中是否包含触发词
        seg_triggers = [t for t in triggers if t in seg]
        if not seg_triggers:
            continue

        # 只连接触发词两侧最近的实体对，而非所有实体对
        for trigger in seg_triggers:
            trigger_pos = seg.find(trigger)
            # 找触发词左侧最近的实体
            left_entities = []
            right_entities = []
            for e in seg_entities:
                e_pos = seg.find(e["name"])
                if e_pos < trigger_pos:
                    left_entities.append((e_pos, e))
                elif e_pos >= trigger_pos + len(trigger):
                    right_entities.append((e_pos, e))

            # 取左侧最近和右侧最近的实体建立关系
            if left_entities and right_entities:
                left_entities.sort(key=lambda x: -x[0])  # 离触发词最近的
                right_entities.sort(key=lambda x: x[0])
                left_e = left_entities[0][1]
                right_e = right_entities[0][1]

                from_e, to_e, from_t, to_t = _determine_relation_direction(
                    left_e, right_e, relation_type, seg
                )
                # 跳过不合理的类型组合
                if not from_e:
                    continue
                key = f"{from_e}|{relation_type}|{to_e}"
                if key not in seen:
                    seen.add(key)
                    triples.append({
                        "from_entity": from_e,
                        "from_type": from_t,
                        "relation": relation_type,
                        "to_entity": to_e,
                        "to_type": to_t,
                    })

    return triples


def _determine_relation_direction(
    e1: Dict, e2: Dict, relation: str, segment: str
) -> Tuple[str, str, str, str]:
    """根据关系类型和实体类型确定关系方向，并过滤不合理的组合

    Returns: (from_entity, to_entity, from_type, to_type)
           返回空字符串表示该关系不合理
    """
    n1, t1 = e1["name"], e1["type"]
    n2, t2 = e2["name"], e2["type"]

    if relation == "causes":
        # 因果方向：Component -> Fault, Fault -> Fault, Component -> Component
        if t1 == "Component" and t2 == "Fault":
            return n1, n2, t1, t2
        elif t1 == "Fault" and t2 == "Component":
            return n2, n1, t2, t1
        elif t1 == "Fault" and t2 == "Fault":
            return n1, n2, t1, t2
        elif t1 == "Component" and t2 == "Component":
            return n1, n2, t1, t2
        return "", "", "", ""

    elif relation == "fixes":
        # 修复方向：Command/Config -> Fault
        if t1 in ("Command", "Config") and t2 == "Fault":
            return n1, n2, t1, t2
        elif t1 == "Fault" and t2 in ("Command", "Config"):
            return n2, n1, t2, t1
        return "", "", "", ""

    elif relation == "configures":
        # 配置方向：Config -> Component, Config -> Fault
        if t1 == "Config" and t2 in ("Component", "Fault"):
            return n1, n2, t1, t2
        elif t1 in ("Component", "Fault") and t2 == "Config":
            return n2, n1, t2, t1
        return "", "", "", ""

    elif relation == "checks":
        # 检查方向：Command -> Component, Command -> Fault, Command -> Config
        if t1 == "Command" and t2 in ("Component", "Fault", "Config"):
            return n1, n2, t1, t2
        elif t1 in ("Component", "Fault", "Config") and t2 == "Command":
            return n2, n1, t2, t1
        return "", "", "", ""

    elif relation == "monitors":
        # 监控方向：Command -> Component, Component -> Metric
        if t1 == "Command" and t2 in ("Component", "Metric"):
            return n1, n2, t1, t2
        elif t1 in ("Component", "Metric") and t2 == "Command":
            return n2, n1, t2, t1
        elif t1 == "Component" and t2 == "Metric":
            return n1, n2, t1, t2
        elif t1 == "Metric" and t2 == "Component":
            return n2, n1, t2, t1
        return "", "", "", ""

    elif relation == "indicates":
        # 指示方向：Metric -> Fault
        if t1 == "Metric" and t2 == "Fault":
            return n1, n2, t1, t2
        elif t1 == "Fault" and t2 == "Metric":
            return n2, n1, t2, t1
        return "", "", "", ""

    elif relation == "depends_on":
        # 依赖方向：Component -> Component
        if t1 == "Component" and t2 == "Component":
            return n1, n2, t1, t2
        return "", "", "", ""

    return "", "", "", ""


def _infer_entity_type(name: str) -> str:
    """根据名称推断实体类型"""
    # 检查是否是组件
    for comp_name, pattern in COMPONENT_PATTERNS.items():
        if re.fullmatch(pattern, name, re.IGNORECASE):
            return "Component"
    # 检查是否是故障
    for pattern, fault_name in FAULT_PATTERNS:
        if fault_name == name:
            return "Fault"
    # 检查是否是命令
    for cmd_pattern, _ in COMMAND_PATTERNS:
        if re.fullmatch(cmd_pattern, name, re.IGNORECASE):
            return "Command"
    # 检查是否是配置项
    for cfg_pattern, _ in CONFIG_PATTERNS:
        if re.search(cfg_pattern, name, re.IGNORECASE):
            return "Config"
    return "Component"


def extract_causal_relations(text: str) -> List[Dict]:
    """抽取因果关系"""
    return _match_relation_patterns(text, CAUSAL_PATTERNS, "causes")


def extract_fix_relations(text: str) -> List[Dict]:
    """抽取修复关系"""
    return _match_relation_patterns(text, FIX_PATTERNS, "fixes")


def extract_configure_relations(text: str) -> List[Dict]:
    """抽取配置关系"""
    return _match_relation_patterns(text, CONFIGURE_PATTERNS, "configures")


def extract_dependency_relations(text: str) -> List[Dict]:
    """抽取依赖关系"""
    return _match_relation_patterns(text, DEPEND_PATTERNS, "depends_on")


def extract_monitor_relations(text: str) -> List[Dict]:
    """抽取监控关系"""
    return _match_relation_patterns(text, MONITOR_PATTERNS, "monitors")


def extract_check_relations(text: str) -> List[Dict]:
    """抽取检查关系"""
    return _match_relation_patterns(text, CHECK_PATTERNS, "checks")


def extract_indicate_relations(text: str) -> List[Dict]:
    """抽取指示关系"""
    return _match_relation_patterns(text, INDICATE_PATTERNS, "indicates")


def extract_component_config_relations(text: str, entities: List[Dict]) -> List[Dict]:
    """根据实体共现抽取组件-配置项关系

    如果文本中同时出现某个组件和其对应的配置项，建立 configures 关系
    """
    triples = []
    configs = [e for e in entities if e["type"] == "Config"]
    components = [e for e in entities if e["type"] == "Component"]

    # 配置项与组件的映射
    config_component_map = {}
    for pattern, comp in CONFIG_PATTERNS:
        config_name = pattern.split(r'\s')[0].replace(r'(?:', '').replace(r')?', '')
        config_component_map[config_name] = comp

    for cfg in configs:
        # 优先使用映射关系
        mapped_comp = config_component_map.get(cfg["name"])
        if mapped_comp:
            # 检查该组件是否在文本中出现
            comp_names = [c["name"] for c in components]
            if mapped_comp in comp_names:
                triples.append({
                    "from_entity": cfg["name"],
                    "from_type": "Config",
                    "relation": "configures",
                    "to_entity": mapped_comp,
                    "to_type": "Component",
                })
        else:
            # 回退：与同文本中出现的组件建立关系
            for comp in components:
                triples.append({
                    "from_entity": cfg["name"],
                    "from_type": "Config",
                    "relation": "configures",
                    "to_entity": comp["name"],
                    "to_type": "Component",
                })
    return triples


def extract_component_fault_relations(text: str, entities: List[Dict]) -> List[Dict]:
    """根据实体共现和上下文抽取组件-故障关系

    如果文本中同时出现某个组件和故障，且上下文暗示因果，建立 causes 关系
    """
    triples = []
    faults = [e for e in entities if e["type"] == "Fault"]
    components = [e for e in entities if e["type"] == "Component"]

    # 检查"现象"关键词上下文
    symptom_keywords = ["现象", "问题", "报错", "错误", "异常", "出现"]
    for fault in faults:
        for comp in components:
            # 检查故障名附近是否提到组件
            comp_name = comp["name"]
            fault_name = fault["name"]
            pattern = re.compile(
                rf'\b{re.escape(comp_name)}\b.*?{re.escape(fault_name)}|'
                rf'{re.escape(fault_name)}.*?\b{re.escape(comp_name)}\b',
                re.IGNORECASE
            )
            if pattern.search(text):
                # 进一步检查是否有因果暗示
                has_causal = any(kw in text for kw in symptom_keywords)
                if has_causal:
                    triples.append({
                        "from_entity": comp["name"],
                        "from_type": "Component",
                        "relation": "causes",
                        "to_entity": fault["name"],
                        "to_type": "Fault",
                    })
    return triples


def extract_command_check_relations(text: str, entities: List[Dict]) -> List[Dict]:
    """抽取命令-组件检查关系

    如果文本中命令和组件同时出现，且上下文暗示检查/查看，建立 checks 关系
    """
    triples = []
    commands = [e for e in entities if e["type"] == "Command"]
    components = [e for e in entities if e["type"] == "Component"]

    check_keywords = ["检查", "查看", "验证", "确认", "监控", "测试", "排查"]
    for cmd in commands:
        for comp in components:
            # 检查命令和组件是否在同一段落/句子中出现
            pattern = re.compile(
                rf'{re.escape(cmd["name"])}.*?{re.escape(comp["name"])}|'
                rf'{re.escape(comp["name"])}.*?{re.escape(cmd["name"])}',
                re.IGNORECASE
            )
            if pattern.search(text):
                has_check = any(kw in text for kw in check_keywords)
                if has_check:
                    triples.append({
                        "from_entity": cmd["name"],
                        "from_type": "Command",
                        "relation": "checks",
                        "to_entity": comp["name"],
                        "to_type": "Component",
                    })
    return triples


def extract_triples(text: str) -> List[Dict]:
    """使用规则模板从文本中抽取所有三元组

    Returns:
        三元组列表，每个元素包含 from_entity, from_type, relation, to_entity, to_type
    """
    all_triples = []
    seen = set()

    # 1. 基于模式匹配的关系抽取
    pattern_extractors = [
        extract_causal_relations,
        extract_fix_relations,
        extract_configure_relations,
        extract_dependency_relations,
        extract_monitor_relations,
        extract_check_relations,
        extract_indicate_relations,
    ]
    for extractor in pattern_extractors:
        for triple in extractor(text):
            key = f"{triple['from_entity']}|{triple['relation']}|{triple['to_entity']}"
            if key not in seen:
                seen.add(key)
                all_triples.append(triple)

    # 2. 基于实体共现的关系抽取
    entities = extract_all_entities(text)

    for triple in extract_component_config_relations(text, entities):
        key = f"{triple['from_entity']}|{triple['relation']}|{triple['to_entity']}"
        if key not in seen:
            seen.add(key)
            all_triples.append(triple)

    for triple in extract_component_fault_relations(text, entities):
        key = f"{triple['from_entity']}|{triple['relation']}|{triple['to_entity']}"
        if key not in seen:
            seen.add(key)
            all_triples.append(triple)

    for triple in extract_command_check_relations(text, entities):
        key = f"{triple['from_entity']}|{triple['relation']}|{triple['to_entity']}"
        if key not in seen:
            seen.add(key)
            all_triples.append(triple)

    if all_triples:
        logger.info(f"[规则抽取] 从文本中抽取 {len(all_triples)} 个三元组")
    return all_triples
