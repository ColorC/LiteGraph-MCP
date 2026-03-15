# -*- coding: utf-8 -*-
"""
轻量级图数据库管理器 (Lightweight Graph Manager)

纯 SQLite 实现，按需查询，零内存开销。
外部进程写入 SQLite 后立即可见。
"""

import json
import sqlite3
import logging
import random
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Iterator

logger = logging.getLogger(__name__)


class LightweightGraphManager:
    """轻量级图管理器：纯 SQLite，按需查询"""

    def __init__(self, db_path: str = "data/kg_graph.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        nc, ec = self.node_count(), self.edge_count()
        logger.info(f"[Graph] 初始化完成: {nc} 节点, {ec} 边 ({self.db_path})")

    @contextmanager
    def _conn(self):
        """SQLite 连接上下文管理器"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self):
        """初始化SQLite表结构"""
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS nodes (
                    id TEXT PRIMARY KEY,
                    label TEXT,
                    properties JSON
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS edges (
                    source TEXT,
                    target TEXT,
                    relationship TEXT,
                    properties JSON,
                    PRIMARY KEY (source, target, relationship)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_node_label ON nodes(label)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_edge_rel ON edges(relationship)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_edge_source ON edges(source)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_edge_target ON edges(target)")

    def _parse_node(self, row) -> Dict[str, Any]:
        """将 SQLite 行解析为节点字典"""
        props = json.loads(row["properties"]) if row["properties"] else {}
        props.pop("label", None)
        return {"id": row["id"], "label": row["label"], **props}

    def _parse_edge(self, row) -> Dict[str, Any]:
        """将 SQLite 行解析为边字典"""
        props = json.loads(row["properties"]) if row["properties"] else {}
        return {"source": row["source"], "target": row["target"],
                "relationship": row["relationship"], **props}

    # -------------------------------------------------------------------------
    # 便捷查询方法
    # -------------------------------------------------------------------------

    def has_node(self, node_id: str) -> bool:
        with self._conn() as conn:
            row = conn.execute("SELECT 1 FROM nodes WHERE id=? LIMIT 1", (node_id,)).fetchone()
            return row is not None

    def has_edge(self, source: str, target: str, relationship: str = None) -> bool:
        with self._conn() as conn:
            if relationship:
                row = conn.execute(
                    "SELECT 1 FROM edges WHERE source=? AND target=? AND relationship=? LIMIT 1",
                    (source, target, relationship)).fetchone()
            else:
                row = conn.execute(
                    "SELECT 1 FROM edges WHERE source=? AND target=? LIMIT 1",
                    (source, target)).fetchone()
            return row is not None

    def node_count(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]

    def edge_count(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]

    # -------------------------------------------------------------------------
    # CRUD Operations
    # -------------------------------------------------------------------------

    def add_node(self, node_id: str, label: str, properties: Dict[str, Any] = None) -> bool:
        props = properties or {}
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO nodes (id, label, properties) VALUES (?, ?, ?)",
                (node_id, label, json.dumps(props, ensure_ascii=False)))
        return True

    def delete_node(self, node_id: str) -> bool:
        with self._conn() as conn:
            conn.execute("DELETE FROM edges WHERE source=? OR target=?", (node_id, node_id))
            conn.execute("DELETE FROM nodes WHERE id=?", (node_id,))
        return True

    def add_edge(self, source: str, target: str, relationship: str, properties: Dict[str, Any] = None) -> bool:
        props = properties or {}
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO edges (source, target, relationship, properties) VALUES (?, ?, ?, ?)",
                (source, target, relationship, json.dumps(props, ensure_ascii=False)))
        return True

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            row = conn.execute("SELECT id, label, properties FROM nodes WHERE id=?", (node_id,)).fetchone()
            if row:
                return self._parse_node(row)
            return None

    def find_nodes_by_label(self, label: str) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute("SELECT id, label, properties FROM nodes WHERE label=?", (label,)).fetchall()
            return [self._parse_node(r) for r in rows]

    def iter_all_nodes(self) -> Iterator[Dict[str, Any]]:
        """流式迭代所有节点（避免一次性加载全表到内存）"""
        with self._conn() as conn:
            cursor = conn.execute("SELECT id, label, properties FROM nodes")
            for row in cursor:
                yield self._parse_node(row)

    def iter_nodes_by_label(self, label: str) -> Iterator[Dict[str, Any]]:
        """流式迭代指定标签节点"""
        with self._conn() as conn:
            cursor = conn.execute(
                "SELECT id, label, properties FROM nodes WHERE label=?",
                (label,),
            )
            for row in cursor:
                yield self._parse_node(row)

    def find_related_nodes(self, node_id: str, relationship: str = None, direction: str = "OUT") -> List[Dict[str, Any]]:
        results = []
        with self._conn() as conn:
            if direction in ("OUT", "BOTH"):
                if relationship:
                    rows = conn.execute(
                        "SELECT n.id, n.label, n.properties FROM edges e JOIN nodes n ON e.target=n.id "
                        "WHERE e.source=? AND e.relationship=?", (node_id, relationship)).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT n.id, n.label, n.properties FROM edges e JOIN nodes n ON e.target=n.id "
                        "WHERE e.source=?", (node_id,)).fetchall()
                results.extend(self._parse_node(r) for r in rows)

            if direction in ("IN", "BOTH"):
                if relationship:
                    rows = conn.execute(
                        "SELECT n.id, n.label, n.properties FROM edges e JOIN nodes n ON e.source=n.id "
                        "WHERE e.target=? AND e.relationship=?", (node_id, relationship)).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT n.id, n.label, n.properties FROM edges e JOIN nodes n ON e.source=n.id "
                        "WHERE e.target=?", (node_id,)).fetchall()
                results.extend(self._parse_node(r) for r in rows)
        return results

    def find_edges(self, node_id: str, direction: str = "OUT", relationship: str = None) -> List[Dict[str, Any]]:
        results = []
        with self._conn() as conn:
            if direction in ("OUT", "BOTH"):
                if relationship:
                    rows = conn.execute(
                        "SELECT source, target, relationship, properties FROM edges WHERE source=? AND relationship=?",
                        (node_id, relationship)).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT source, target, relationship, properties FROM edges WHERE source=?",
                        (node_id,)).fetchall()
                results.extend(self._parse_edge(r) for r in rows)

            if direction in ("IN", "BOTH"):
                if relationship:
                    rows = conn.execute(
                        "SELECT source, target, relationship, properties FROM edges WHERE target=? AND relationship=?",
                        (node_id, relationship)).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT source, target, relationship, properties FROM edges WHERE target=?",
                        (node_id,)).fetchall()
                results.extend(self._parse_edge(r) for r in rows)
        return results

    def get_edge(self, source: str, target: str, relationship: str = None) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            if relationship:
                row = conn.execute(
                    "SELECT source, target, relationship, properties FROM edges "
                    "WHERE source=? AND target=? AND relationship=?",
                    (source, target, relationship)).fetchone()
            else:
                row = conn.execute(
                    "SELECT source, target, relationship, properties FROM edges "
                    "WHERE source=? AND target=?",
                    (source, target)).fetchone()
            if row:
                return self._parse_edge(row)
            return None

    def delete_edge(self, source: str, target: str, relationship: str = None) -> bool:
        with self._conn() as conn:
            if relationship:
                conn.execute("DELETE FROM edges WHERE source=? AND target=? AND relationship=?",
                             (source, target, relationship))
            else:
                conn.execute("DELETE FROM edges WHERE source=? AND target=?", (source, target))
        return True

    def update_node_properties(self, node_id: str, props: Dict[str, Any]) -> bool:
        """合并更新节点属性"""
        with self._conn() as conn:
            row = conn.execute("SELECT properties FROM nodes WHERE id=?", (node_id,)).fetchone()
            if not row:
                return False
            existing = json.loads(row["properties"]) if row["properties"] else {}
            existing.update(props)
            conn.execute("UPDATE nodes SET properties=? WHERE id=?",
                         (json.dumps(existing, ensure_ascii=False), node_id))
        return True

    def update_node(self, node_id: str, props: Dict[str, Any]) -> bool:
        """更新节点属性（兼容别名）"""
        return self.update_node_properties(node_id, props)

    # -------------------------------------------------------------------------
    # 搜索
    # -------------------------------------------------------------------------

    def search_nodes(self, keyword: str, limit: int = 20) -> List[Dict[str, Any]]:
        """按关键词搜索节点（匹配 id、label、name）"""
        pattern = f"%{keyword}%"
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, label, properties FROM nodes "
                "WHERE id LIKE ? OR label LIKE ? OR "
                "json_extract(properties, '$.name') LIKE ? "
                "LIMIT ?",
                (pattern, pattern, pattern, limit)).fetchall()
            return [self._parse_node(r) for r in rows]

    def get_all_labels(self) -> List[str]:
        """获取所有节点标签"""
        with self._conn() as conn:
            rows = conn.execute("SELECT DISTINCT label FROM nodes ORDER BY label").fetchall()
            return [r["label"] for r in rows if r["label"]]

    def get_node_types_with_counts(self) -> List[Dict[str, Any]]:
        """获取所有节点类型及其数量"""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT label, COUNT(*) as count FROM nodes WHERE label IS NOT NULL GROUP BY label ORDER BY count DESC"
            ).fetchall()
            return [{"label": r["label"], "count": r["count"]} for r in rows if r["label"]]

    # -------------------------------------------------------------------------
    # 图采样
    # -------------------------------------------------------------------------

    def random_nodes(self, limit: int = 50, labels: List[str] = None) -> List[Dict[str, Any]]:
        """随机获取节点，可按标签过滤"""
        with self._conn() as conn:
            if labels:
                placeholders = ",".join("?" * len(labels))
                rows = conn.execute(
                    f"SELECT id, label, properties FROM nodes WHERE label IN ({placeholders}) ORDER BY RANDOM() LIMIT ?",
                    (*labels, limit)).fetchall()
            else:
                rows = conn.execute("SELECT id, label, properties FROM nodes ORDER BY RANDOM() LIMIT ?", (limit,)).fetchall()
            return [self._parse_node(r) for r in rows]

    def get_random_subgraph(self, limit: int = 50, labels: List[str] = None) -> Dict[str, Any]:
        """网格状采样：随机种子 + 逐条扩展边。支持标签过滤，优先网状扩展，扩展不动时再加散点。"""
        with self._conn() as conn:
            # 随机种子（优先从指定标签中选）
            if labels:
                placeholders = ",".join("?" * len(labels))
                seed_row = conn.execute(
                    f"SELECT id FROM nodes WHERE label IN ({placeholders}) ORDER BY RANDOM() LIMIT 1",
                    tuple(labels)).fetchone()
            else:
                seed_row = conn.execute("SELECT id FROM nodes ORDER BY RANDOM() LIMIT 1").fetchone()

            if not seed_row:
                return {"nodes": [], "edges": []}

            visited = {seed_row["id"]}
            edges = []

            # 阶段1：网状扩展
            stale_rounds = 0
            for _ in range(limit * 3):  # 多尝试几轮
                if len(visited) >= limit:
                    break

                candidates = list(visited)
                random.shuffle(candidates)
                expanded = False
                for node in candidates:
                    row = conn.execute(
                        "SELECT source, target, relationship, properties FROM edges "
                        "WHERE (source=? OR target=?) ORDER BY RANDOM() LIMIT 20",
                        (node, node)).fetchall()
                    for edge_row in row:
                        neighbor = edge_row["target"] if edge_row["source"] == node else edge_row["source"]
                        if neighbor in visited:
                            continue
                        # 如果有标签过滤，检查邻居标签
                        if labels:
                            n_row = conn.execute("SELECT label FROM nodes WHERE id=?", (neighbor,)).fetchone()
                            if not n_row or n_row["label"] not in labels:
                                continue
                        visited.add(neighbor)
                        edges.append(self._parse_edge(edge_row))
                        expanded = True
                        break
                    if expanded:
                        break

                if not expanded:
                    stale_rounds += 1
                    if stale_rounds >= 3:
                        # 网状扩展已死，跳到新种子继续尝试
                        if labels:
                            placeholders_v = ",".join("?" * len(visited))
                            placeholders_l = ",".join("?" * len(labels))
                            new_seed = conn.execute(
                                f"SELECT id FROM nodes WHERE id NOT IN ({placeholders_v}) AND label IN ({placeholders_l}) ORDER BY RANDOM() LIMIT 1",
                                (*visited, *labels)).fetchone()
                        else:
                            placeholders_v = ",".join("?" * len(visited))
                            new_seed = conn.execute(
                                f"SELECT id FROM nodes WHERE id NOT IN ({placeholders_v}) ORDER BY RANDOM() LIMIT 1",
                                tuple(visited)).fetchone()
                        if not new_seed:
                            break
                        visited.add(new_seed["id"])
                        stale_rounds = 0
                else:
                    stale_rounds = 0

            # 获取所有节点数据
            nodes_data = []
            for nid in visited:
                nrow = conn.execute("SELECT id, label, properties FROM nodes WHERE id=?", (nid,)).fetchone()
                if nrow:
                    nodes_data.append(self._parse_node(nrow))

            return {"nodes": nodes_data, "edges": edges}

    # -------------------------------------------------------------------------
    # 图算法（多跳遍历）
    # -------------------------------------------------------------------------

    def get_neighbors(
        self,
        node_id: str,
        max_hops: int = 2,
        relationship: str = None,
        skip_infra: bool = True,
        max_size: int = 200,
    ) -> Dict[str, Any]:
        """N跳邻域查询，使用 SQL 递归 CTE"""
        with self._conn() as conn:
            visited_nodes = set()
            visited_edges = []
            frontier = {node_id}

            for hop in range(max_hops):
                if not frontier or len(visited_nodes) >= max_size:
                    break
                next_frontier = set()
                for nid in frontier:
                    # 出边
                    if relationship:
                        out_rows = conn.execute(
                            "SELECT source, target, relationship, properties FROM edges "
                            "WHERE source=? AND relationship=?", (nid, relationship)).fetchall()
                    else:
                        out_rows = conn.execute(
                            "SELECT source, target, relationship, properties FROM edges WHERE source=?",
                            (nid,)).fetchall()
                    # 入边
                    if relationship:
                        in_rows = conn.execute(
                            "SELECT source, target, relationship, properties FROM edges "
                            "WHERE target=? AND relationship=?", (nid, relationship)).fetchall()
                    else:
                        in_rows = conn.execute(
                            "SELECT source, target, relationship, properties FROM edges WHERE target=?",
                            (nid,)).fetchall()

                    for row in list(out_rows) + list(in_rows):
                        neighbor = row["target"] if row["source"] == nid else row["source"]
                        if neighbor in visited_nodes or neighbor in frontier:
                            continue
                        # infra 过滤
                        if skip_infra:
                            n_row = conn.execute("SELECT properties FROM nodes WHERE id=?", (neighbor,)).fetchone()
                            if n_row:
                                n_props = json.loads(n_row["properties"]) if n_row["properties"] else {}
                                if n_props.get("infra_layer") == "L1":
                                    continue
                        next_frontier.add(neighbor)
                        visited_edges.append(self._parse_edge(row))
                        if len(visited_nodes) + len(next_frontier) >= max_size:
                            break

                visited_nodes.update(frontier)
                frontier = next_frontier

            visited_nodes.update(frontier)

            # 获取节点数据
            nodes_data = []
            for nid in visited_nodes:
                nrow = conn.execute("SELECT id, label, properties FROM nodes WHERE id=?", (nid,)).fetchone()
                if nrow:
                    nodes_data.append(self._parse_node(nrow))

            return {"nodes": nodes_data, "edges": visited_edges}

    def compute_ppr(self, seed_nodes: List[str], alpha: float = 0.85, top_k: int = 50) -> List[Tuple[str, float]]:
        """
        近似 Personalized PageRank（多轮 BFS 权重衰减）。
        不依赖 NetworkX，用 SQL 查询邻居实现。
        """
        scores: Dict[str, float] = {}
        for seed in seed_nodes:
            scores[seed] = 1.0 / len(seed_nodes)

        with self._conn() as conn:
            for _ in range(10):  # 10 轮迭代
                new_scores: Dict[str, float] = {}
                for nid, score in scores.items():
                    # teleport
                    if nid in [s for s in seed_nodes]:
                        new_scores[nid] = new_scores.get(nid, 0) + (1 - alpha) * (1.0 / len(seed_nodes))

                    # 扩散到邻居
                    out_rows = conn.execute("SELECT target FROM edges WHERE source=?", (nid,)).fetchall()
                    in_rows = conn.execute("SELECT source FROM edges WHERE target=?", (nid,)).fetchall()
                    neighbors = [r[0] for r in out_rows] + [r[0] for r in in_rows]
                    if neighbors:
                        spread = alpha * score / len(neighbors)
                        for nb in neighbors:
                            new_scores[nb] = new_scores.get(nb, 0) + spread

                scores = new_scores

        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return ranked[:top_k]

    # -------------------------------------------------------------------------
    # 高级操作
    # -------------------------------------------------------------------------

    def execute_basic_query(self, query_type: str, params: Dict[str, Any]) -> List[Any]:
        if query_type == "find_by_label":
            return self.find_nodes_by_label(params.get("label", ""))
        elif query_type == "find_related":
            return self.find_related_nodes(
                params.get("node_id", ""),
                relationship=params.get("relationship"),
                direction=params.get("direction", "OUT"))
        elif query_type == "shortest_path":
            return self._shortest_path(params.get("source", ""), params.get("target", ""))
        return []

    def _shortest_path(self, source: str, target: str, max_depth: int = 10) -> List[str]:
        """BFS 最短路径"""
        if source == target:
            return [source]
        with self._conn() as conn:
            visited = {source}
            parent = {source: None}
            queue = [source]
            for _ in range(max_depth):
                if not queue:
                    break
                next_queue = []
                for nid in queue:
                    rows = conn.execute(
                        "SELECT target FROM edges WHERE source=? "
                        "UNION SELECT source FROM edges WHERE target=?",
                        (nid, nid)).fetchall()
                    for row in rows:
                        nb = row[0]
                        if nb not in visited:
                            visited.add(nb)
                            parent[nb] = nid
                            if nb == target:
                                # 回溯路径
                                path = [target]
                                cur = target
                                while parent[cur] is not None:
                                    cur = parent[cur]
                                    path.append(cur)
                                return list(reversed(path))
                            next_queue.append(nb)
                queue = next_queue
        return []

    def shortest_path_detail(self, source: str, target: str, max_depth: int = 6) -> Dict[str, Any]:
        """带边信息的最短路径"""
        path_ids = self._shortest_path(source, target, max_depth)
        if not path_ids:
            return {"found": False, "path": [], "edges": []}
        nodes = []
        edges = []
        with self._conn() as conn:
            for nid in path_ids:
                row = conn.execute("SELECT id, label, properties FROM nodes WHERE id=?", (nid,)).fetchone()
                if row:
                    parsed = self._parse_node(row)
                    parsed.pop("_embedding", None)
                    nodes.append({"id": parsed["id"], "label": parsed["label"], "name": parsed.get("name", parsed["id"])})
            for i in range(len(path_ids) - 1):
                a, b = path_ids[i], path_ids[i + 1]
                row = conn.execute(
                    "SELECT source, target, relationship FROM edges WHERE (source=? AND target=?) OR (source=? AND target=?)",
                    (a, b, b, a)).fetchone()
                if row:
                    edges.append({"source": row["source"], "target": row["target"], "relationship": row["relationship"]})
        return {"found": True, "hops": len(path_ids) - 1, "path": nodes, "edges": edges}

    def extract_subgraph(self, seed_ids: List[str], max_hops: int = 2, max_nodes: int = 50) -> Dict[str, Any]:
        """提取种子节点周围的子图，合并每个种子的 N-hop 邻域"""
        valid_seeds = [s for s in seed_ids if self.has_node(s)]
        if not valid_seeds:
            return {"node_count": 0, "edge_count": 0, "nodes": [], "edges": []}

        per_seed = max(10, max_nodes // max(len(valid_seeds), 1))
        all_node_ids = set(valid_seeds)
        all_edges_raw = []
        seen_edge_keys = set()

        for seed in valid_seeds:
            nb = self.get_neighbors(seed, max_hops=max_hops, max_size=per_seed)
            for n in nb.get("nodes", []):
                all_node_ids.add(n["id"])
            for e in nb.get("edges", []):
                key = (e.get("source"), e.get("target"), e.get("relationship"))
                if key not in seen_edge_keys:
                    seen_edge_keys.add(key)
                    all_edges_raw.append(e)

        # 种子优先，截断
        ordered = list(valid_seeds) + [nid for nid in all_node_ids if nid not in valid_seeds]
        ordered = ordered[:max_nodes]
        final_set = set(ordered)

        # 只保留两端都在 final_set 的边
        final_edges = [
            {"source": e["source"], "target": e["target"], "relationship": e["relationship"]}
            for e in all_edges_raw
            if e.get("source") in final_set and e.get("target") in final_set
        ]

        with self._conn() as conn:
            nodes = []
            for nid in ordered:
                row = conn.execute("SELECT id, label, properties FROM nodes WHERE id=?", (nid,)).fetchone()
                if row:
                    parsed = self._parse_node(row)
                    parsed.pop("_embedding", None)
                    nodes.append({"id": parsed["id"], "label": parsed["label"],
                                  "name": parsed.get("name", parsed["id"]),
                                  "description": parsed.get("description", "")[:200]})

        return {"node_count": len(nodes), "edge_count": len(final_edges), "nodes": nodes, "edges": final_edges}

    def get_schema_info(self, detail: str = "overview") -> Dict[str, Any]:
        """图结构 schema 信息"""
        with self._conn() as conn:
            if detail == "overview":
                label_rows = conn.execute(
                    "SELECT label, COUNT(*) as cnt FROM nodes GROUP BY label ORDER BY cnt DESC").fetchall()
                rel_rows = conn.execute(
                    "SELECT relationship, COUNT(*) as cnt FROM edges GROUP BY relationship ORDER BY cnt DESC").fetchall()
                return {
                    "node_labels": {r["label"]: r["cnt"] for r in label_rows},
                    "edge_types": {r["relationship"]: r["cnt"] for r in rel_rows},
                }
            elif detail == "relationships":
                rows = conn.execute("""
                    SELECT e.relationship,
                           n1.label as src_label, n2.label as tgt_label,
                           COUNT(*) as cnt
                    FROM edges e
                    JOIN nodes n1 ON e.source = n1.id
                    JOIN nodes n2 ON e.target = n2.id
                    GROUP BY e.relationship, n1.label, n2.label
                    ORDER BY cnt DESC
                """).fetchall()
                patterns = []
                for r in rows:
                    patterns.append({
                        "pattern": f"{r['src_label']} -[{r['relationship']}]-> {r['tgt_label']}",
                        "count": r["cnt"],
                    })
                return {"relationship_patterns": patterns}
            elif detail == "label_detail":
                label_rows = conn.execute(
                    "SELECT label, COUNT(*) as cnt FROM nodes GROUP BY label ORDER BY cnt DESC").fetchall()
                result = {}
                for lr in label_rows:
                    sample = conn.execute(
                        "SELECT properties FROM nodes WHERE label=? LIMIT 1", (lr["label"],)).fetchone()
                    if sample and sample["properties"]:
                        props = json.loads(sample["properties"])
                        fields = [k for k in props.keys() if not k.startswith("_")]
                        result[lr["label"]] = {"count": lr["cnt"], "fields": fields}
                    else:
                        result[lr["label"]] = {"count": lr["cnt"], "fields": []}
                return {"labels": result}
            return {}

    def traverse(self, start_id: str, edge_types: List[str], direction: str = "OUT",
                 max_hops: int = 3, max_results: int = 30) -> Dict[str, Any]:
        """按指定关系类型链式遍历"""
        with self._conn() as conn:
            layers = []
            frontier = {start_id}
            visited = {start_id}
            for hop in range(max_hops):
                if not frontier:
                    break
                layer_nodes = []
                layer_edges = []
                next_frontier = set()
                for nid in frontier:
                    for etype in edge_types:
                        if direction == "OUT":
                            rows = conn.execute(
                                "SELECT source, target, relationship FROM edges WHERE source=? AND relationship=?",
                                (nid, etype)).fetchall()
                        elif direction == "IN":
                            rows = conn.execute(
                                "SELECT source, target, relationship FROM edges WHERE target=? AND relationship=?",
                                (nid, etype)).fetchall()
                        else:  # BOTH
                            rows = conn.execute(
                                "SELECT source, target, relationship FROM edges WHERE (source=? OR target=?) AND relationship=?",
                                (nid, nid, etype)).fetchall()
                        for r in rows:
                            neighbor = r["target"] if r["source"] == nid else r["source"]
                            if neighbor not in visited and len(visited) + len(next_frontier) < max_results:
                                next_frontier.add(neighbor)
                                nrow = conn.execute("SELECT id, label, properties FROM nodes WHERE id=?", (neighbor,)).fetchone()
                                if nrow:
                                    parsed = self._parse_node(nrow)
                                    layer_nodes.append({"id": parsed["id"], "label": parsed["label"],
                                                        "name": parsed.get("name", parsed["id"])})
                                layer_edges.append({"source": r["source"], "target": r["target"], "relationship": r["relationship"]})
                if layer_nodes:
                    layers.append({"hop": hop + 1, "nodes": layer_nodes, "edges": layer_edges})
                visited.update(next_frontier)
                frontier = next_frontier
            return {"start": start_id, "total_reached": len(visited) - 1, "layers": layers}

    def merge_nodes(
        self,
        source_ids: List[str],
        target_id: str,
        delete_sources: bool = False,
    ) -> Dict[str, Any]:
        """合并节点：将 source 节点的边重定向到 target"""
        redirected = 0
        with self._conn() as conn:
            for sid in source_ids:
                if sid == target_id:
                    continue
                # 重定向出边
                out_rows = conn.execute(
                    "SELECT source, target, relationship, properties FROM edges WHERE source=?",
                    (sid,)).fetchall()
                for row in out_rows:
                    if row["target"] != target_id:
                        props = row["properties"] or "{}"
                        conn.execute(
                            "INSERT OR IGNORE INTO edges (source, target, relationship, properties) VALUES (?, ?, ?, ?)",
                            (target_id, row["target"], row["relationship"], props))
                        redirected += 1

                # 重定向入边
                in_rows = conn.execute(
                    "SELECT source, target, relationship, properties FROM edges WHERE target=?",
                    (sid,)).fetchall()
                for row in in_rows:
                    if row["source"] != target_id:
                        props = row["properties"] or "{}"
                        conn.execute(
                            "INSERT OR IGNORE INTO edges (source, target, relationship, properties) VALUES (?, ?, ?, ?)",
                            (row["source"], target_id, row["relationship"], props))
                        redirected += 1

                # 删除旧边
                conn.execute("DELETE FROM edges WHERE source=? OR target=?", (sid, sid))

                if delete_sources:
                    conn.execute("DELETE FROM nodes WHERE id=?", (sid,))

        return {"redirected_edges": redirected, "deleted_sources": delete_sources}

    def _is_infra_node(self, node_id: str) -> bool:
        node = self.get_node(node_id)
        if node:
            return node.get("infra_layer") == "L1"
        return False

    # -------------------------------------------------------------------------
    # 批量操作
    # -------------------------------------------------------------------------

    def batch_ingest(self, nodes: List[tuple], edges: List[tuple], chunk_size: int = 5000) -> Dict[str, int]:
        """批量导入节点和边"""
        with self._conn() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            node_count = 0
            for i in range(0, len(nodes), chunk_size):
                chunk = nodes[i:i + chunk_size]
                conn.executemany(
                    "INSERT OR REPLACE INTO nodes (id, label, properties) VALUES (?, ?, ?)",
                    [(n_id, label, json.dumps(props, ensure_ascii=False)) for n_id, label, props in chunk])
                conn.commit()
                node_count += len(chunk)
                if node_count % 10000 == 0 or node_count == len(nodes):
                    logger.info(f"[Graph] 节点写入进度: {node_count}/{len(nodes)}")

            edge_count = 0
            for i in range(0, len(edges), chunk_size):
                chunk = edges[i:i + chunk_size]
                conn.executemany(
                    "INSERT OR REPLACE INTO edges (source, target, relationship, properties) VALUES (?, ?, ?, ?)",
                    [(src, tgt, rel, json.dumps(props, ensure_ascii=False)) for src, tgt, rel, props in chunk])
                conn.commit()
                edge_count += len(chunk)
                if edge_count % 10000 == 0 or edge_count == len(edges):
                    logger.info(f"[Graph] 边写入进度: {edge_count}/{len(edges)}")

        logger.info(f"[Graph] 批量导入完成: {node_count} 节点, {edge_count} 边")
        return {"nodes_added": node_count, "edges_added": edge_count}

    def clear(self):
        """清空图数据库"""
        with self._conn() as conn:
            conn.execute("DELETE FROM nodes")
            conn.execute("DELETE FROM edges")
        logger.info("[Graph] 数据库已清空")

    def close(self):
        pass
