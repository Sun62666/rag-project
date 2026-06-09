"""知识图谱可视化辅助函数"""
import logging

logger = logging.getLogger(__name__)

_COLOR_PALETTE = [
    "GroupA", "GroupB", "GroupC", "GroupD", "GroupE", "GroupF", "GroupG", "GroupH"
]


def pick_label(labels: list) -> str:
    """从 Neo4j 节点标签列表中选出最有意义的标签"""
    priority = ["Component", "Fault", "Command", "Config", "Metric", "Service", "Protocol"]
    for p in priority:
        if p in labels:
            return p
    if "Entity" in labels and len(labels) == 1:
        return "Entity"
    for l in labels:
        if l != "Entity":
            return l
    return "Entity"


def pick_label_str(label_str: str) -> str:
    """检查单个标签字符串是否为已知运维实体类型"""
    priority = ["Component", "Fault", "Command", "Config", "Metric", "Service", "Protocol"]
    if label_str in priority:
        return label_str
    return "Entity"


def assign_color_groups(nodes_set: dict, edges_list: list, center_entity: str = None):
    """为未分类的 Entity 节点分配颜色组"""
    entity_nodes = [nid for nid, ndata in nodes_set.items() if ndata["group"] == "Entity"]
    if not entity_nodes:
        return
    all_nodes = list(nodes_set.keys())
    neighbor_map = {nid: set() for nid in all_nodes}
    for e in edges_list:
        f, t = e["from"], e["to"]
        if f in neighbor_map:
            neighbor_map[f].add(t)
        if t in neighbor_map:
            neighbor_map[t].add(f)

    if center_entity and center_entity in nodes_set:
        dist = {center_entity: 0}
        queue = [center_entity]
        while queue:
            cur = queue.pop(0)
            for nb in neighbor_map.get(cur, []):
                if nb not in dist:
                    dist[nb] = dist[cur] + 1
                    queue.append(nb)
        for nid in entity_nodes:
            d = dist.get(nid, 999)
            nodes_set[nid]["group"] = _COLOR_PALETTE[d % len(_COLOR_PALETTE)]
        return

    assigned = {}
    for nid in entity_nodes:
        if nid in assigned:
            continue
        group_idx = len(assigned) % len(_COLOR_PALETTE)
        group_name = _COLOR_PALETTE[group_idx]
        stack = [nid]
        while stack:
            cur = stack.pop()
            if cur in assigned:
                continue
            assigned[cur] = group_name
            for nb in neighbor_map.get(cur, []):
                if nb in nodes_set and nodes_set[nb]["group"] == "Entity" and nb not in assigned:
                    stack.append(nb)
    for nid, group_name in assigned.items():
        nodes_set[nid]["group"] = group_name


def build_vis_data(kg, entity_name: str, depth: int) -> dict:
    """构建以指定实体为中心的可视化数据"""
    nodes_set = {}
    edges_list = []
    seen_edge_keys = set()
    _visited = set()

    def _traverse(name: str, current_depth: int):
        if current_depth > depth or name in _visited:
            return
        _visited.add(name)
        records = kg.query_related(name, depth=1)
        for r in records:
            src, src_type = r["source"], r["source_type"]
            tgt, tgt_type = r["target"], r["target_type"]
            rel = r["relation"]

            if src not in nodes_set:
                nodes_set[src] = {"id": src, "label": src, "group": pick_label_str(src_type) if src_type else "Entity"}
            if tgt not in nodes_set:
                nodes_set[tgt] = {"id": tgt, "label": tgt, "group": pick_label_str(tgt_type) if tgt_type else "Entity"}

            edge_key = f"{src}-{rel}-{tgt}"
            if edge_key not in seen_edge_keys:
                seen_edge_keys.add(edge_key)
                edges_list.append({
                    "id": edge_key,
                    "from": src,
                    "to": tgt,
                    "label": rel,
                    "arrows": "to",
                })

            if current_depth < depth:
                _traverse(tgt, current_depth + 1)
                _traverse(src, current_depth + 1)

    nodes_set[entity_name] = {"id": entity_name, "label": entity_name, "group": "Query"}
    _traverse(entity_name, 1)

    exact_match = len(nodes_set) > 1

    if len(nodes_set) <= 1:
        with kg._driver.session() as session:
            fuzzy = session.run(
                "MATCH (n) WHERE n.name CONTAINS $keyword RETURN n.name AS name, labels(n) AS types LIMIT 10",
                keyword=entity_name,
            )
            for record in fuzzy:
                name = record["name"]
                ntype = pick_label(record["types"]) if record["types"] else "Entity"
                if name not in nodes_set:
                    nodes_set[name] = {"id": name, "label": name, "group": ntype}
                    _traverse(name, 1)

    assign_color_groups(nodes_set, edges_list, center_entity=entity_name if exact_match else None)

    return {
        "available": True,
        "nodes": list(nodes_set.values()),
        "edges": edges_list,
    }


def build_full_vis_data(kg) -> dict:
    """构建全图概览的可视化数据"""
    if not kg.is_available:
        return {"available": False, "nodes": [], "edges": []}
    try:
        nodes_set = {}
        edges_list = []
        seen_edge_keys = set()
        with kg._driver.session() as session:
            result = session.run(
                "MATCH (n)-[r]->(m) "
                "RETURN n.name AS src, labels(n) AS src_type, "
                "type(r) AS rel, m.name AS tgt, labels(m) AS tgt_type "
                "LIMIT 300"
            )
            for record in result:
                src = record["src"]
                tgt = record["tgt"]
                src_labels = record["src_type"] or ["Entity"]
                tgt_labels = record["tgt_type"] or ["Entity"]
                src_type = pick_label(src_labels)
                tgt_type = pick_label(tgt_labels)
                rel = record["rel"]

                if src not in nodes_set:
                    nodes_set[src] = {"id": src, "label": src, "group": src_type}
                if tgt not in nodes_set:
                    nodes_set[tgt] = {"id": tgt, "label": tgt, "group": tgt_type}

                edge_key = f"{src}-{rel}-{tgt}"
                if edge_key not in seen_edge_keys:
                    seen_edge_keys.add(edge_key)
                    edges_list.append({
                        "id": edge_key,
                        "from": src,
                        "to": tgt,
                        "label": rel,
                        "arrows": "to",
                    })

            if not nodes_set:
                solo_result = session.run(
                    "MATCH (n) RETURN n.name AS name, labels(n) AS types LIMIT 100"
                )
                for record in solo_result:
                    name = record["name"]
                    ntype = pick_label(record["types"]) if record["types"] else "Entity"
                    if name and name not in nodes_set:
                        nodes_set[name] = {"id": name, "label": name, "group": ntype}

        assign_color_groups(nodes_set, edges_list)

        return {
            "available": True,
            "nodes": list(nodes_set.values()),
            "edges": edges_list,
        }
    except Exception as e:
        logger.error(f"[知识图谱] 构建vis数据失败: {e}")
        return {"available": False, "error": str(e), "nodes": [], "edges": []}
