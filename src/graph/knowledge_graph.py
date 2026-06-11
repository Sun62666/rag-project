import logging
import re
import json
from typing import List, Dict, Optional
from src.core.config import get_settings

logger = logging.getLogger(__name__)

ENTITY_TYPES = [
    "Component", "Fault", "Command", "Config", "Metric", "Service", "Protocol"
]

RELATION_TYPES = [
    "causes", "fixes", "depends_on", "monitors", "configures",
    "relates_to", "indicates", "restarts", "checks"
]


class OpsKnowledgeGraph:
    """运维知识图谱：基于 Neo4j 的实体关系图，支持三种抽取方式、图查询、多跳推理

    抽取方式：
    1. rule: 基于正则和模式匹配的规则模板抽取（快速、无需模型）
    2. spacy: 基于 spaCy NLP 的依存句法分析抽取（中等速度、需下载模型）
    3. llm: 基于大语言模型的深度理解抽取（慢但最准确）
    """

    def __init__(self):
        self.cfg = get_settings()
        self._driver = None
        self._connected = False
        self._init_neo4j()

    def _init_neo4j(self):
        try:
            from neo4j import GraphDatabase
            uri = self.cfg.NEO4J_URI
            user = self.cfg.NEO4J_USER
            password = self.cfg.NEO4J_PASSWORD
            if uri:
                try:
                    self._driver = GraphDatabase.driver(uri, auth=(user, password))
                    self._driver.verify_connectivity()
                    self._connected = True
                    self._create_constraints()
                    logger.info(f"[知识图谱] Neo4j 连接成功: {uri}")
                except Exception as e:
                    logger.warning(f"[知识图谱] Neo4j 连接失败，知识图谱不可用: {e}")
                    self._connected = False
            else:
                logger.info("[知识图谱] NEO4J_URI 未配置，知识图谱不可用")
                self._connected = False
        except ImportError:
            logger.warning("[知识图谱] neo4j 包未安装，请运行 pip install neo4j")
            self._connected = False

    def _create_constraints(self):
        if not self._connected:
            return
        with self._driver.session() as session:
            for entity_type in ENTITY_TYPES:
                session.run(
                    f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{entity_type}) REQUIRE n.name IS UNIQUE"
                )

    @property
    def is_available(self) -> bool:
        return self._connected

    def add_entity(self, name: str, entity_type: str, properties: Optional[Dict] = None):
        if not self._connected:
            return
        props = properties or {}
        props["name"] = name
        props_str = ", ".join([f"{k}: ${k}" for k in props.keys()])
        query = f"MERGE (e:{entity_type} {{name: $name}}) SET e += {{{props_str}}}"
        try:
            with self._driver.session() as session:
                session.run(query, **props)
        except Exception as e:
            logger.error(f"[知识图谱] 添加实体失败: {e}")

    def add_relation(self, from_name: str, from_type: str, relation: str, to_name: str, to_type: str):
        if not self._connected:
            return
        query = (
            f"MATCH (a:{from_type} {{name: $from_name}}) "
            f"MATCH (b:{to_type} {{name: $to_name}}) "
            f"MERGE (a)-[r:{relation}]->(b)"
        )
        try:
            with self._driver.session() as session:
                session.run(query, from_name=from_name, to_name=to_name)
        except Exception as e:
            logger.error(f"[知识图谱] 添加关系失败: {e}")

    def add_triple(self, from_entity: str, from_type: str, relation: str, to_entity: str, to_type: str):
        self.add_entity(from_entity, from_type)
        self.add_entity(to_entity, to_type)
        self.add_relation(from_entity, from_type, relation, to_entity, to_type)

    def add_triples(self, triples: List[Dict]) -> int:
        """批量写入三元组，返回成功写入的数量"""
        if not self._connected:
            return 0
        count = 0
        for t in triples:
            try:
                self.add_triple(
                    t["from_entity"], t["from_type"],
                    t["relation"],
                    t["to_entity"], t["to_type"],
                )
                count += 1
            except Exception as e:
                logger.warning(f"[知识图谱] 写入三元组失败: {e}")
        return count

    # ========== 三种抽取方式 ==========

    def extract_with_rules(self, text: str) -> List[Dict]:
        """方式1: 规则模板抽取（快速、无需模型）"""
        from src.graph.extractors.rule_extractor import extract_triples
        return extract_triples(text)

    def extract_with_spacy(self, text: str) -> List[Dict]:
        """方式2: spaCy NLP 抽取（中等速度、需下载模型）"""
        from src.graph.extractors.spacy_extractor import extract_triples
        return extract_triples(text)

    def extract_with_llm(self, text: str) -> List[Dict]:
        """方式3: LLM 大模型抽取（慢但最准确）"""
        from src.graph.extractors.llm_extractor import extract_triples
        return extract_triples(text)

    def extract_triples(self, text: str, method: str = "hybrid") -> List[Dict]:
        """抽取三元组，支持三种方式和混合模式

        Args:
            text: 运维文档文本
            method: 抽取方式
                - "rule": 仅规则模板
                - "spacy": 仅 spaCy
                - "llm": 仅 LLM
                - "hybrid": 混合模式（推荐），先规则+spaCy快速抽取，再用LLM补充

        Returns:
            三元组列表
        """
        if method == "rule":
            return self.extract_with_rules(text)
        elif method == "spacy":
            return self.extract_with_spacy(text)
        elif method == "llm":
            return self.extract_with_llm(text)
        elif method == "hybrid":
            return self._extract_hybrid(text)
        else:
            logger.warning(f"[知识图谱] 未知抽取方式: {method}，使用 hybrid")
            return self._extract_hybrid(text)

    def _extract_hybrid(self, text: str) -> List[Dict]:
        """混合抽取：规则 + spaCy 快速抽取，LLM 补充

        策略：
        1. 先用规则模板快速抽取（零成本）
        2. 再用 spaCy 抽取（低成本）
        3. 合并去重后，如果数量不足再用 LLM 补充
        """
        all_triples = []
        seen_keys = set()

        def _add_triples(triples: List[Dict], source: str):
            for t in triples:
                key = f"{t['from_entity']}|{t['relation']}|{t['to_entity']}"
                if key not in seen_keys:
                    seen_keys.add(key)
                    all_triples.append(t)

        # Step 1: 规则模板抽取
        rule_triples = self.extract_with_rules(text)
        _add_triples(rule_triples, "rule")
        logger.info(f"[混合抽取] 规则抽取: {len(rule_triples)} 个")

        # Step 2: spaCy 抽取
        spacy_triples = self.extract_with_spacy(text)
        _add_triples(spacy_triples, "spacy")
        logger.info(f"[混合抽取] spaCy 抽取: {len(spacy_triples)} 个（新增后共 {len(all_triples)} 个）")

        # Step 3: 如果规则+spaCy抽取不足，用 LLM 补充
        if len(all_triples) < 15:
            llm_triples = self.extract_with_llm(text)
            _add_triples(llm_triples, "llm")
            logger.info(f"[混合抽取] LLM 补充: {len(llm_triples)} 个（新增后共 {len(all_triples)} 个）")

        logger.info(f"[混合抽取] 最终结果: {len(all_triples)} 个三元组")
        return all_triples

    def extract_and_ingest(self, text: str, source: str = "", method: str = "hybrid") -> int:
        """从文本中抽取三元组并写入知识图谱

        Args:
            text: 运维文档文本
            source: 数据来源标识
            method: 抽取方式 ("rule", "spacy", "llm", "hybrid")
        """
        triples = self.extract_triples(text, method=method)
        if not triples:
            return 0
        count = self.add_triples(triples)
        logger.info(f"[知识图谱] 从 '{source}' 抽取并写入 {count} 个三元组 (方式: {method})")
        return count

    # ========== 图查询 ==========

    def query_related(self, entity_name: str, depth: int = 2) -> List[Dict]:
        if not self._connected:
            return []
        query = (
            f"MATCH path = (e {{name: $name}})-[r*1..{depth}]-(n) "
            f"UNWIND relationships(path) AS rel "
            f"RETURN DISTINCT startNode(rel).name AS source, "
            f"labels(startNode(rel))[0] AS source_type, "
            f"type(rel) AS relation, "
            f"endNode(rel).name AS target, "
            f"labels(endNode(rel))[0] AS target_type"
        )
        try:
            with self._driver.session() as session:
                result = session.run(query, name=entity_name)
                records = []
                for record in result:
                    if not record["source"] or not record["target"]:
                        continue
                    records.append({
                        "source": record["source"],
                        "source_type": record["source_type"] or "",
                        "relation": record["relation"],
                        "target": record["target"],
                        "target_type": record["target_type"] or "",
                    })
                # 精确匹配无结果时，尝试模糊匹配
                if not records:
                    records = self._fuzzy_query_related(entity_name, depth)
                return records
        except Exception as e:
            logger.error(f"[知识图谱] 查询失败: {e}")
            try:
                simple_query = (
                    "MATCH (e {name: $name})-[r]-(n) "
                    "RETURN e.name AS source, labels(e)[0] AS source_type, "
                    "type(r) AS relation, n.name AS target, labels(n)[0] AS target_type"
                )
                with self._driver.session() as session:
                    result = session.run(simple_query, name=entity_name)
                    records = []
                    for record in result:
                        records.append({
                            "source": record["source"],
                            "source_type": record["source_type"] or "",
                            "relation": record["relation"],
                            "target": record["target"],
                            "target_type": record["target_type"] or "",
                        })
                    if not records:
                        return self._fuzzy_query_related(entity_name, depth)
                    return records
            except Exception as e2:
                logger.error(f"[知识图谱] 降级查询也失败: {e2}")
                return []

    def _fuzzy_query_related(self, entity_name: str, depth: int = 1) -> List[Dict]:
        """模糊匹配查询：当精确匹配无结果时，用 CONTAINS 查找相似实体"""
        if not self._connected:
            return []
        try:
            with self._driver.session() as session:
                # 先找到模糊匹配的实体名
                fuzzy_result = session.run(
                    "MATCH (n) WHERE n.name CONTAINS $name OR $name CONTAINS n.name "
                    "RETURN n.name AS name LIMIT 5",
                    name=entity_name,
                )
                matched_names = [r["name"] for r in fuzzy_result if r["name"]]
                if not matched_names:
                    return []
                logger.info(f"[知识图谱] 模糊匹配 '{entity_name}' -> {matched_names}")
                # 对匹配到的实体查询关联关系
                all_records = []
                seen = set()
                for name in matched_names:
                    query = (
                        f"MATCH path = (e {{name: $name}})-[r*1..{depth}]-(n) "
                        f"UNWIND relationships(path) AS rel "
                        f"RETURN DISTINCT startNode(rel).name AS source, "
                        f"labels(startNode(rel))[0] AS source_type, "
                        f"type(rel) AS relation, "
                        f"endNode(rel).name AS target, "
                        f"labels(endNode(rel))[0] AS target_type"
                    )
                    result = session.run(query, name=name)
                    for record in result:
                        if not record["source"] or not record["target"]:
                            continue
                        key = f"{record['source']}-{record['relation']}-{record['target']}"
                        if key not in seen:
                            seen.add(key)
                            all_records.append({
                                "source": record["source"],
                                "source_type": record["source_type"] or "",
                                "relation": record["relation"],
                                "target": record["target"],
                                "target_type": record["target_type"] or "",
                            })
                return all_records
        except Exception as e:
            logger.debug(f"[知识图谱] 模糊查询失败: {e}")
            return []

    def query_fault_chain(self, fault_name: str) -> List[Dict]:
        """查询故障链路：包括故障的原因链和影响链

        方向1: (Component/Config) --[causes]--> Fault  （什么导致了故障）
        方向2: Fault --[causes]--> Fault/Component     （故障导致了什么）
        """
        if not self._connected:
            return []
        results = []
        try:
            with self._driver.session() as session:
                # 查询1: 什么导致了该故障（反向追溯）
                cause_query = (
                    "MATCH (cause)-[r:causes]->(f:Fault {name: $name}) "
                    "RETURN cause.name AS from_entity, labels(cause)[0] AS from_type, "
                    "type(r) AS relation, f.name AS to_entity, 'Fault' AS to_type"
                )
                result = session.run(cause_query, name=fault_name)
                for record in result:
                    results.append(dict(record))

                # 查询2: 该故障导致了什么（正向传播）
                impact_query = (
                    "MATCH (f:Fault {name: $name})-[r:causes]->(impact) "
                    "RETURN f.name AS from_entity, 'Fault' AS from_type, "
                    "type(r) AS relation, impact.name AS to_entity, labels(impact)[0] AS to_type"
                )
                result = session.run(impact_query, name=fault_name)
                for record in result:
                    results.append(dict(record))

                # 查询3: 多跳因果链（Component -> Fault -> Fault）
                chain_query = (
                    "MATCH path = (cause)-[:causes*1..3]->(f:Fault {name: $name}) "
                    "UNWIND relationships(path) AS r "
                    "RETURN DISTINCT startNode(r).name AS from_entity, "
                    "labels(startNode(r))[0] AS from_type, "
                    "type(r) AS relation, "
                    "endNode(r).name AS to_entity, "
                    "labels(endNode(r))[0] AS to_type"
                )
                result = session.run(chain_query, name=fault_name)
                for record in result:
                    r = dict(record)
                    if r not in results:
                        results.append(r)
        except Exception as e:
            logger.error(f"[知识图谱] 故障链查询失败: {e}")
        return results

    def query_fix_for_fault(self, fault_name: str) -> List[Dict]:
        """查询故障的修复方案

        方向: (Command/Config) --[fixes]--> Fault
        同时也查 configures 关系（配置项可以间接修复故障）
        """
        if not self._connected:
            return []
        results = []
        try:
            with self._driver.session() as session:
                # 查询1: 直接修复关系
                fix_query = (
                    "MATCH (fix)-[r:fixes]->(f:Fault {name: $name}) "
                    "RETURN fix.name AS fix_name, labels(fix)[0] AS fix_type, type(r) AS relation"
                )
                result = session.run(fix_query, name=fault_name)
                for record in result:
                    results.append(dict(record))

                # 查询2: 通过配置项间接修复（configures 关系）
                config_fix_query = (
                    "MATCH (config:Config)-[:configures]->(comp:Component)-[:causes]->(f:Fault {name: $name}) "
                    "RETURN config.name AS fix_name, 'Config' AS fix_type, 'configures->causes' AS relation"
                )
                result = session.run(config_fix_query, name=fault_name)
                for record in result:
                    results.append(dict(record))

                # 查询3: 模糊匹配（故障名部分匹配）
                if not results:
                    fuzzy_query = (
                        "MATCH (fix)-[r:fixes]->(f:Fault) "
                        "WHERE f.name CONTAINS $name OR $name CONTAINS f.name "
                        "RETURN fix.name AS fix_name, labels(fix)[0] AS fix_type, type(r) AS relation"
                    )
                    result = session.run(fuzzy_query, name=fault_name)
                    for record in result:
                        results.append(dict(record))
        except Exception as e:
            logger.error(f"[知识图谱] 修复方案查询失败: {e}")
        return results

    # ========== 上下文生成 ==========

    def format_graph_context(self, query: str, depth: int = 2) -> str:
        ops_entities = self._extract_entities_from_query(query)
        if not ops_entities:
            return ""

        all_records = []
        for entity_name in ops_entities:
            records = self.query_related(entity_name, depth)
            all_records.extend(records)

        if not all_records:
            return ""

        seen = set()
        unique = []
        for r in all_records:
            key = f"{r['source']}-{r['relation']}-{r['target']}"
            if key not in seen:
                seen.add(key)
                unique.append(r)

        parts = ["【知识图谱关联信息】"]
        for r in unique[:15]:
            parts.append(f"  {r['source']}({r['source_type']}) --[{r['relation']}]--> {r['target']}({r['target_type']})")

        fault_chains = []
        fix_results = []
        for entity_name in ops_entities:
            # 查故障链（包含原因链 + 影响链 + 多跳因果链）
            chain = self.query_fault_chain(entity_name)
            if chain:
                fault_chains.extend(chain)

            # 查修复方案（包含直接修复 + 配置间接修复 + 模糊匹配）
            fixes = self.query_fix_for_fault(entity_name)
            if fixes:
                fix_results.extend(fixes)

        if fault_chains:
            parts.append("\n【故障影响链路】")
            seen_chains = set()
            for fc in fault_chains[:10]:
                chain_key = f"{fc.get('from_entity','')}-{fc.get('relation','')}-{fc.get('to_entity','')}"
                if chain_key not in seen_chains:
                    seen_chains.add(chain_key)
                    from_t = fc.get('from_type', '')
                    to_t = fc.get('to_type', '')
                    parts.append(f"  {fc['from_entity']}({from_t}) --[{fc['relation']}]--> {fc['to_entity']}({to_t})")

        if fix_results:
            parts.append("\n【关联修复方案】")
            for fx in fix_results[:10]:
                fix_type = fx.get("fix_type", [""])[0] if fx.get("fix_type") else ""
                parts.append(f"  {fx['fix_name']}({fix_type})")

        context = "\n".join(parts)
        logger.info(f"[知识图谱] 生成上下文: {len(unique)} 个关系, {len(fault_chains)} 条故障链, {len(fix_results)} 个修复方案")
        return context

    def _extract_entities_from_query(self, query: str) -> List[str]:
        """从用户问题中抽取实体名称，用于知识图谱查询

        采用三级抽取策略：
        1. 规则模板快速匹配（零延迟）
        2. 图数据库索引匹配（精确，利用已有实体）
        3. spaCy NLP 抽取（中等延迟，处理自然语言表述）
        """
        found = []
        seen = set()

        def _add(name: str):
            name = name.strip()
            if name and name not in seen:
                seen.add(name)
                found.append(name)

        # ---- 级别1: 规则模板快速匹配（复用 rule_extractor 的实体模式） ----
        try:
            from src.graph.extractors.rule_extractor import extract_components, extract_faults
            for e in extract_components(query):
                _add(e["name"])
            for e in extract_faults(query):
                _add(e["name"])
        except Exception:
            # 降级：硬编码兜底
            known_components = [
                "Redis", "MySQL", "Nginx", "Docker", "Kubernetes", "K8s",
                "Linux", "CentOS", "Ubuntu", "Tomcat", "Apache", "Kafka",
                "RabbitMQ", "Elasticsearch", "MongoDB", "PostgreSQL",
                "Prometheus", "Grafana", "Jenkins", "GitLab", "etcd",
                "Zookeeper", "Hadoop", "Spark", "Flink",
            ]
            known_faults = [
                "OOM", "OutOfMemory", "超时", "timeout", "连接失败",
                "CPU满载", "CPU过高", "磁盘满", "内存溢出", "内存泄漏",
                "连接数满", "端口占用", "服务不可用", "502", "503", "500",
            ]
            query_lower = query.lower()
            for comp in known_components:
                if comp.lower() in query_lower:
                    _add(comp)
            for fault in known_faults:
                if fault.lower() in query_lower:
                    _add(fault)

        # ---- 级别2: 图数据库索引匹配（查询 Neo4j 中已有的实体名） ----
        if self._connected:
            try:
                with self._driver.session() as session:
                    # 模糊匹配：用户问题中包含图中的实体名
                    result = session.run("MATCH (n) RETURN n.name AS name")
                    for record in result:
                        name = record["name"]
                        if name and name in query:
                            _add(name)
                        # 反向匹配：实体名包含在用户问题中（处理简称）
                        elif name and len(name) > 1 and name.lower() in query.lower():
                            _add(name)
            except Exception as e:
                logger.debug(f"[知识图谱] 图索引匹配失败: {e}")

        # ---- 级别3: spaCy NLP 抽取（处理自然语言表述） ----
        if len(found) < 2:  # 如果前两级已找到足够实体，跳过 spaCy
            try:
                from src.graph.extractors.spacy_extractor import extract_entities
                spacy_entities = extract_entities(query)
                for e in spacy_entities:
                    _add(e["name"])
            except Exception:
                pass

        logger.info(f"[知识图谱] 从问题中抽取实体: {found}")
        return found

    # ========== 统计 ==========

    def get_stats(self) -> Dict:
        if not self._connected:
            return {"available": False}
        try:
            with self._driver.session() as session:
                node_count = session.run("MATCH (n) RETURN count(n) AS cnt").single()["cnt"]
                rel_count = session.run("MATCH ()-[r]->() RETURN count(r) AS cnt").single()["cnt"]
                type_counts = {}
                for et in ENTITY_TYPES:
                    result = session.run(f"MATCH (n:{et}) RETURN count(n) AS cnt")
                    cnt = result.single()["cnt"]
                    if cnt > 0:
                        type_counts[et] = cnt
                return {
                    "available": True,
                    "total_nodes": node_count,
                    "total_relations": rel_count,
                    "entity_types": type_counts,
                }
        except Exception as e:
            return {"available": False, "error": str(e)}

    def close(self):
        if self._driver:
            self._driver.close()
            self._connected = False
            logger.info("[知识图谱] Neo4j 连接已关闭")


_kg_instance: Optional["OpsKnowledgeGraph"] = None


def get_knowledge_graph() -> OpsKnowledgeGraph:
    global _kg_instance
    if _kg_instance is None:
        _kg_instance = OpsKnowledgeGraph()
    return _kg_instance
