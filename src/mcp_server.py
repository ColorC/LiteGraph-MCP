#!/usr/bin/env python3
"""
OpenGraph Graph RAG MCP Server (HTTP)

提供图谱查询和编辑的 MCP 工具，通过 HTTP 协议暴露。
启动后可以通过 MCP 客户端连接 http://localhost:8001/mcp

用法:
    python -m src.mcp_server --port 8001 --host 0.0.0.0
"""

import argparse
import json
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any, Optional

# 添加项目根目录到 path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mcp.server.fastmcp import FastMCP

from src.graph.manager import LightweightGraphManager
from src.tools.question_manager import QuestionManager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 全局变量
gm: Optional[LightweightGraphManager] = None
qm: Optional[QuestionManager] = None
retriever = None  # GraphRAGRetriever 实例，按需初始化

# 创建 FastMCP 实例（host/port 会在 main() 中通过 settings 覆盖）
mcp = FastMCP(
    name="OpenGraph Graph RAG",
    instructions="游戏知识图谱查询和编辑服务。支持图谱搜索、邻居查询、节点编辑等操作。",
    host="0.0.0.0",
    port=8001,
    transport_security={"enable_dns_rebinding_protection": False},
)


def init_shared(shared_gm, shared_retriever, shared_qm):
    """注入共享实例（兼容旧统一入口调用）"""
    global gm, qm, retriever
    gm = shared_gm
    retriever = shared_retriever
    qm = shared_qm
    logger.info("MCP HTTP 服务已注入共享实例")


def _ensure_retriever():
    """按需初始化检索器，避免服务启动即加载大索引。"""
    global retriever
    if retriever is not None:
        return retriever

    from src.retrieval.hybrid_retriever import GraphRAGRetriever

    snippet_db_path = str(ROOT / "data" / "code_snippets.db")
    retriever = GraphRAGRetriever(gm, snippet_db_path=snippet_db_path)
    logger.info("Hybrid retriever 已按需初始化")
    return retriever


def init_graph(db_path: str = "data/kg_graph.db"):
    """独立初始化图谱管理器（fallback: 直接运行本文件时使用）"""
    global gm, qm, retriever
    db_file = ROOT / db_path
    if not db_file.exists():
        # 尝试其他路径
        db_file = ROOT / "data" / "open_graph_graph.db"
    if not db_file.exists():
        raise FileNotFoundError(f"图谱数据库不存在: {db_file}")

    gm = LightweightGraphManager(str(db_file))
    qm = QuestionManager(gm)
    logger.info(f"图谱数据库已加载: {db_file}")

    snippet_db_path = str(ROOT / "data" / "code_snippets.db")

    # 后台启动片段增量更新（检索器按需初始化，不在启动时加载）
    _start_snippet_updater(str(db_file), snippet_db_path)


def _start_snippet_updater(main_db_path: str, snippet_db_path: str):
    """在后台线程中运行片段增量更新"""
    bridge_url = os.environ.get("WINDOWS_FILE_BRIDGE_URL", "")
    if not bridge_url:
        logger.info("[snippet_updater] WINDOWS_FILE_BRIDGE_URL 未设置，跳过片段索引更新")
        return

    def _run():
        try:
            from src.indexing.snippet_updater import SnippetUpdater
            updater = SnippetUpdater(
                main_db_path=main_db_path,
                snippet_db_path=snippet_db_path,
                bridge_url=bridge_url,
            )
            stats = updater.run()
            logger.info(f"[snippet_updater] 后台更新完成: {stats}")

            # 通知 retriever 重新加载片段索引
            if retriever and stats.get("indexed", 0) > 0:
                retriever.reload_snippet_index()
        except Exception as e:
            logger.error(f"[snippet_updater] 后台更新失败: {e}", exc_info=True)

    thread = threading.Thread(target=_run, name="snippet-updater", daemon=True)
    thread.start()
    logger.info("[snippet_updater] 后台增量更新线程已启动")


# ============ 辅助路由 ============

@mcp.app.get("/health")
async def health_check():
    """健康检查接口，供 open_graph-bridge 探测服务状态。"""
    if gm is None:
        return {"status": "starting", "message": "Graph manager not initialized"}
    return {
        "status": "healthy",
        "nodes": gm.node_count(),
        "edges": gm.edge_count()
    }

# ============ 工具定义 ============

def _clamp_int(value: Any, default: int, min_value: int, max_value: int) -> int:
    try:
        iv = int(value)
    except Exception:
        iv = default
    return max(min_value, min(max_value, iv))


def _safe_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, default=str)


def _strip_embedding_in_nodes(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned = []
    for node in nodes:
        item = dict(node)
        item.pop("_embedding", None)
        cleaned.append(item)
    return cleaned


def _extract_code_asset_link(entry: dict[str, Any]) -> str:
    node_id = str(entry.get("id", ""))
    node = gm.get_node(node_id) if gm else None
    if not node:
        return node_id

    path = str(node.get("path", "")).replace("\\", "/")
    if path:
        for prefix in ("C:/Git/record/", "C:\\Git\\record\\", "c:/git/record/", "c:\\git\\record\\"):
            if path.lower().startswith(prefix.lower()):
                return f"record/{path[len(prefix):]}".replace("//", "/")
        return f"record/{path}".replace("//", "/")

    if node_id.startswith("file:"):
        return f"record/{node_id[5:]}".replace("//", "/")
    return node_id


def _generate_repo_map_for_codes(code_index: list[dict[str, Any]], max_files: int, timeout_ms: int) -> tuple[dict[str, Any], str, dict[str, Any]]:
    if not code_index:
        return {}, "ok", {"generated": 0, "requested": 0}

    bridge_url = os.environ.get("WINDOWS_FILE_BRIDGE_URL", "").strip()
    if not bridge_url:
        return {}, "bridge_unavailable", {"generated": 0, "requested": min(len(code_index), max_files)}

    from concurrent.futures import ThreadPoolExecutor, TimeoutError
    from src.tools.repo_map import RepoMapGenerator

    generator = RepoMapGenerator(root_path=".", remote_bridge_url=bridge_url)
    targets = code_index[:max_files]
    requested = len(targets)

    def _run() -> dict[str, Any]:
        out: dict[str, Any] = {}
        for item in targets:
            fp = item.get("asset_link") or item.get("name") or item.get("id")
            sig = generator.generate_for_file(str(fp))
            if sig:
                out[str(fp)] = sig.format(query="")
        return out

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_run)
        try:
            repo_map = future.result(timeout=max(0.1, timeout_ms / 1000.0))
            return repo_map, "ok", {"generated": len(repo_map), "requested": requested}
        except TimeoutError:
            return {}, "timeout", {"generated": 0, "requested": requested, "timeout_ms": timeout_ms}
        except Exception as e:
            return {}, "generation_failed", {"generated": 0, "requested": requested, "error": str(e)}


def _run_query_default(query: str, top_k: int, depth: int, label_filter: str, mode: str = "hybrid") -> str:
    if retriever is None:
        _ensure_retriever()

    top_k = _clamp_int(top_k, 5, 1, 100)
    depth = _clamp_int(depth, 1, 0, 2)

    if depth >= 1:
        result = retriever.deep_retrieve(query=query, top_k=top_k, mode=mode, label_filter=label_filter)
        if depth == 1:
            result["ppr_expanded"] = []
            result["paths"] = []
            result["narrative"] = retriever._build_narrative(result["seeds"], result["neighbors"], [], [], [])
        return result["narrative"]

    results = retriever.retrieve(query=query, top_k=top_k, mode=mode, label_filter=label_filter)
    return _safe_json(results)


@mcp.tool()
def query_default(query: str, top_k: int = 5, depth: int = 1, label_filter: str = "") -> str:
    """默认查询入口：最少参数，直接可用。"""
    if gm is None:
        return _safe_json({"error": "图谱未初始化"})

    try:
        return _run_query_default(query=query, top_k=top_k, depth=depth, label_filter=label_filter, mode="hybrid")
    except Exception as e:
        return _safe_json({"error": str(e)})


@mcp.tool()
def query_code_semantic(
    query: str,
    top_k: int = 5,
    max_files: int = 5,
    include_repo_map: bool = True,
    repo_map_timeout_ms: int = 3500,
) -> str:
    """代码语义查询入口：固定代码索引路径，并可附加 Repo Map。"""
    if gm is None:
        return _safe_json({"error": "图谱未初始化"})

    try:
        if retriever is None:
            _ensure_retriever()

        top_k = _clamp_int(top_k, 5, 1, 100)
        max_files = _clamp_int(max_files, 5, 0, 50)
        repo_map_timeout_ms = _clamp_int(repo_map_timeout_ms, 3500, 100, 15000)

        rows = retriever.retrieve(
            query=query,
            top_k=top_k,
            mode="dense",
            label_filter="CodeFile",
        )

        code_index = []
        for item in rows:
            code_index.append({
                "id": item.get("id"),
                "name": item.get("name"),
                "label": item.get("label"),
                "score": item.get("score", 0.0),
                "asset_link": _extract_code_asset_link(item),
            })

        repo_map: dict[str, Any] = {}
        repo_map_status = "ok"
        repo_map_meta: dict[str, Any] = {"generated": 0, "requested": 0}
        if include_repo_map:
            repo_map, repo_map_status, repo_map_meta = _generate_repo_map_for_codes(
                code_index=code_index,
                max_files=max_files,
                timeout_ms=repo_map_timeout_ms,
            )

        return _safe_json({
            "results": code_index,
            "code_index": code_index,
            "repo_map": repo_map,
            "repo_map_status": repo_map_status,
            "repo_map_meta": repo_map_meta,
            "result_type": "code_search",
        })
    except Exception as e:
        return _safe_json({"error": str(e)})


@mcp.tool()
def query_custom(
    query: str,
    mode: str = "hybrid",
    top_k: int = 8,
    label_filter: str = "",
    depth: int = 2,
    expand_node_ids: str = "[]",
    expand_hops: int = 2,
    expand_max_size: int = 200,
    include_paths: bool = True,
    include_schema: bool = False,
) -> str:
    """自定义查询入口：支持可控检索与图展开。"""
    if gm is None:
        return _safe_json({"error": "图谱未初始化"})

    try:
        if retriever is None:
            _ensure_retriever()

        top_k = _clamp_int(top_k, 8, 1, 100)
        depth = _clamp_int(depth, 2, 0, 2)
        expand_hops = _clamp_int(expand_hops, 2, 1, 3)
        expand_max_size = _clamp_int(expand_max_size, 200, 1, 400)

        if mode == "code_semantic":
            seeds = retriever.retrieve(query=query, top_k=top_k, mode="dense", label_filter="CodeFile")
            retrieval_result: dict[str, Any] = {
                "seeds": seeds,
                "neighbors": [],
                "ppr_expanded": [],
                "paths": [],
                "narrative": "",
            }
        elif depth >= 1:
            retrieval_result = retriever.deep_retrieve(query=query, top_k=top_k, mode=mode, label_filter=label_filter)
            if depth == 1:
                retrieval_result["ppr_expanded"] = []
                retrieval_result["paths"] = []
            seeds = retrieval_result.get("seeds", [])
        else:
            seeds = retriever.retrieve(query=query, top_k=top_k, mode=mode, label_filter=label_filter)
            retrieval_result = {
                "seeds": seeds,
                "neighbors": [],
                "ppr_expanded": [],
                "paths": [],
                "narrative": "",
            }

        parsed_expand_ids = json.loads(expand_node_ids) if expand_node_ids else []
        if not isinstance(parsed_expand_ids, list):
            return _safe_json({"error": "expand_node_ids 必须是 JSON 数组"})

        centers = [str(x) for x in parsed_expand_ids if x]
        if not centers:
            centers = [str(item.get("id")) for item in seeds if item.get("id")][:top_k]

        graph_nodes: dict[str, dict[str, Any]] = {}
        graph_edges: dict[tuple[str, str, str], dict[str, Any]] = {}

        if centers:
            subgraph = gm.extract_subgraph(centers, max_hops=expand_hops, max_nodes=expand_max_size)
            for node in subgraph.get("nodes", []):
                safe_node = dict(node)
                safe_node.pop("_embedding", None)
                graph_nodes[str(safe_node.get("id"))] = safe_node
            for edge in subgraph.get("edges", []):
                key = (str(edge.get("source")), str(edge.get("target")), str(edge.get("relationship")))
                graph_edges[key] = dict(edge)

        expanded_graph = {
            "nodes": list(graph_nodes.values()),
            "edges": list(graph_edges.values()),
            "centers": centers,
            "hops": expand_hops,
        }

        paths = []
        if include_paths and len(centers) >= 2:
            base = centers[0]
            for target in centers[1:6]:
                path_result = gm.shortest_path_detail(base, target, max_depth=6)
                if path_result.get("found"):
                    paths.append(path_result)

        schema = gm.get_schema_info("overview") if include_schema else None

        return _safe_json({
            "results": expanded_graph,
            "expanded_graph": expanded_graph,
            "expansion_meta": {
                "center_count": len(centers),
                "node_count": len(expanded_graph["nodes"]),
                "edge_count": len(expanded_graph["edges"]),
                "hops": expand_hops,
                "max_size": expand_max_size,
            },
            "retrieval": {
                "seeds": _strip_embedding_in_nodes(retrieval_result.get("seeds", [])),
                "neighbors": _strip_embedding_in_nodes(retrieval_result.get("neighbors", [])),
                "ppr_expanded": _strip_embedding_in_nodes(retrieval_result.get("ppr_expanded", [])),
                "narrative": retrieval_result.get("narrative", ""),
            },
            "paths": paths if include_paths else [],
            "schema": schema,
            "result_type": "graph",
        })
    except json.JSONDecodeError:
        return _safe_json({"error": "expand_node_ids 不是合法的 JSON 数组"})
    except Exception as e:
        return _safe_json({"error": str(e)})


@mcp.tool()
def graph_search(query: str, mode: str = "hybrid", top_k: int = 5, label_filter: str = "", depth: int = 1) -> str:
    """
    兼容旧入口：
    - exact/label_list 保持原行为
    - code_semantic 转发 query_code_semantic
    - custom 转发 query_custom
    - 其余模式走默认查询路径
    """
    if gm is None:
        return _safe_json({"error": "图谱未初始化"})

    try:
        if mode == "exact":
            node = gm.get_node(query)
            if node:
                return _safe_json([node])
            return _safe_json([])

        if mode == "label_list":
            labels = gm.get_all_labels()
            return _safe_json({"labels": labels})

        if mode == "code_semantic":
            return query_code_semantic(query=query, top_k=top_k)

        if mode == "custom":
            return query_custom(query=query, mode="hybrid", top_k=top_k, label_filter=label_filter, depth=depth)

        return _run_query_default(query=query, top_k=top_k, depth=depth, label_filter=label_filter, mode=mode)
    except Exception as e:
        return _safe_json({"error": str(e)})


@mcp.tool()
def graph_neighbors(node_id: str, relationship: Optional[str] = None, hops: int = 1) -> str:
    """
    获取节点的 N 跳邻居和边。

    Args:
        node_id: 节点 ID
        relationship: 可选，过滤特定类型的边
        hops: 跳数（1-3），默认 1

    Returns:
        邻居节点和边信息（JSON 格式）
    """
    if gm is None:
        return json.dumps({"error": "图谱未初始化"})

    try:
        hops = max(1, min(3, hops))
        result = gm.get_neighbors(node_id, max_hops=hops, relationship=relationship, max_size=100)
        # 清理 embedding
        for n in result.get("nodes", []):
            n.pop("_embedding", None)
        return json.dumps({
            "node_id": node_id,
            "hops": hops,
            "node_count": len(result.get("nodes", [])),
            "edge_count": len(result.get("edges", [])),
            "nodes": result.get("nodes", []),
            "edges": result.get("edges", []),
        }, ensure_ascii=False, indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@mcp.tool()
def graph_get_node(node_id: str) -> str:
    """
    获取单个节点的详细信息。
    
    Args:
        node_id: 节点 ID
    
    Returns:
        节点属性（JSON 格式）
    """
    if gm is None:
        return json.dumps({"error": "图谱未初始化"})
    
    try:
        node = gm.get_node(node_id)
        if node is None:
            return json.dumps({"error": f"节点不存在: {node_id}"}, ensure_ascii=False)
        return json.dumps(node, ensure_ascii=False, indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@mcp.tool()
def graph_update_node(node_id: str, properties: str) -> str:
    """
    更新节点属性。
    
    Args:
        node_id: 节点 ID
        properties: JSON 格式的属性字典
    
    Returns:
        操作结果
    """
    if gm is None:
        return json.dumps({"error": "图谱未初始化"})
    
    try:
        props = json.loads(properties)
        gm.update_node(node_id, props)
        return json.dumps({"success": True, "node_id": node_id}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@mcp.tool()
def graph_add_edge(source: str, target: str, relationship: str, properties: Optional[str] = None) -> str:
    """
    创建边。
    
    Args:
        source: 源节点 ID
        target: 目标节点 ID
        relationship: 边类型
        properties: 可选，JSON 格式的边属性
    
    Returns:
        操作结果
    """
    if gm is None:
        return json.dumps({"error": "图谱未初始化"})
    
    try:
        props = json.loads(properties) if properties else None
        gm.add_edge(source, target, relationship, props)
        return json.dumps({"success": True, "edge": f"{source} -> {target}"}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@mcp.tool()
def graph_remove_edge(source: str, target: str, relationship: Optional[str] = None) -> str:
    """
    删除边。
    
    Args:
        source: 源节点 ID
        target: 目标节点 ID
        relationship: 可选，边类型
    
    Returns:
        操作结果
    """
    if gm is None:
        return json.dumps({"error": "图谱未初始化"})
    
    try:
        success = gm.delete_edge(source, target, relationship)
        return json.dumps({"success": success}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@mcp.tool()
def graph_find_path(source_id: str, target_id: str, max_depth: int = 6) -> str:
    """
    查找两个节点之间的最短路径，展示它们如何关联。

    Args:
        source_id: 起始节点 ID
        target_id: 目标节点 ID
        max_depth: 最大搜索深度（1-8），默认 6

    Returns:
        路径上的节点和边（JSON 格式），包含每一跳的节点名称和关系类型
    """
    if gm is None:
        return json.dumps({"error": "图谱未初始化"})

    try:
        max_depth = max(1, min(8, max_depth))
        result = gm.shortest_path_detail(source_id, target_id, max_depth)
        return json.dumps(result, ensure_ascii=False, indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@mcp.tool()
def graph_subgraph(seed_ids: str, max_hops: int = 2, max_nodes: int = 50) -> str:
    """
    提取一组节点之间的连通子图，发现它们的共同关联。

    Args:
        seed_ids: 种子节点 ID 列表，JSON 数组格式，如 '["node1", "node2"]'
        max_hops: 从每个种子扩展的最大跳数（1-3），默认 2
        max_nodes: 子图最大节点数，默认 50

    Returns:
        子图的节点和边（JSON 格式）
    """
    if gm is None:
        return json.dumps({"error": "图谱未初始化"})

    try:
        ids = json.loads(seed_ids)
        if not isinstance(ids, list) or len(ids) < 1:
            return json.dumps({"error": "seed_ids 必须是非空 JSON 数组"})
        max_hops = max(1, min(3, max_hops))
        max_nodes = max(1, min(200, max_nodes))
        result = gm.extract_subgraph(ids, max_hops=max_hops, max_nodes=max_nodes)
        return json.dumps(result, ensure_ascii=False, indent=2, default=str)
    except json.JSONDecodeError:
        return json.dumps({"error": "seed_ids 不是合法的 JSON 数组"})
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@mcp.tool()
def graph_schema(detail: str = "overview") -> str:
    """
    查看图谱的结构信息：有哪些节点类型、关系类型、它们如何连接。

    Args:
        detail: 详细程度
            - overview: 节点标签+计数、关系类型+计数
            - relationships: 每种关系的 源标签→目标标签 模式和计数
            - label_detail: 每种标签的属性字段列表

    Returns:
        图谱结构信息（JSON 格式）
    """
    if gm is None:
        return json.dumps({"error": "图谱未初始化"})

    try:
        result = gm.get_schema_info(detail)
        return json.dumps(result, ensure_ascii=False, indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@mcp.tool()
def graph_traverse(start_id: str, edge_types: str, direction: str = "OUT", max_hops: int = 3, max_results: int = 30) -> str:
    """
    从起点出发，沿指定关系类型链式遍历，发现多跳关联。

    Args:
        start_id: 起始节点 ID
        edge_types: 要遍历的关系类型列表，JSON 数组格式，如 '["HAS_ENTITY", "ENTITY_REFS_FILE"]'
        direction: 遍历方向 - OUT(沿出边)/IN(沿入边)/BOTH(双向)
        max_hops: 最大跳数（1-5），默认 3
        max_results: 最大返回节点数，默认 30

    Returns:
        按层级组织的遍历结果（JSON 格式），每层包含该跳发现的节点和边
    """
    if gm is None:
        return json.dumps({"error": "图谱未初始化"})

    try:
        types = json.loads(edge_types)
        if not isinstance(types, list):
            return json.dumps({"error": "edge_types 必须是 JSON 数组"})
        max_hops = max(1, min(5, max_hops))
        max_results = max(1, min(100, max_results))
        result = gm.traverse(start_id, types, direction=direction, max_hops=max_hops, max_results=max_results)
        return json.dumps(result, ensure_ascii=False, indent=2, default=str)
    except json.JSONDecodeError:
        return json.dumps({"error": "edge_types 不是合法的 JSON 数组"})
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


async def run_async(host: str = "0.0.0.0", port: int = 8001):
    """异步启动 MCP HTTP 服务（供 server.py 统一入口调用）"""
    mcp.settings.host = host
    mcp.settings.port = port
    logger.info(f"启动 MCP HTTP 服务器: http://{host}:{port}/mcp")
    await mcp.run_streamable_http_async()


def main():
    """主函数（独立运行 fallback）"""
    parser = argparse.ArgumentParser(description="OpenGraph Graph RAG MCP Server")
    parser.add_argument("--port", type=int, default=8001, help="服务端口")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="监听地址")
    parser.add_argument("--db", type=str, default="data/kg_graph.db", help="图谱数据库路径")
    args = parser.parse_args()

    # 初始化图谱
    init_graph(args.db)

    # 更新 MCP 配置（FastMCP 从 settings 读取 host/port）
    mcp.settings.host = args.host
    mcp.settings.port = args.port

    logger.info(f"启动 MCP HTTP 服务器: http://{args.host}:{args.port}/mcp")

    # 运行 streamable HTTP 服务器
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
