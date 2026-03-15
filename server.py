#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
open_graph GraphRAG MCP Server

基于 FastMCP 的知识图谱查询和构建服务器。
通过 stdio 模式与 open_graph_agent 通信。

运行:
    python server.py
    或者通过 open_graph_agent config.json5 注册后自动启动
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

# 确保项目根在 sys.path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

logger = logging.getLogger(__name__)

# 创建 MCP 服务器
server = Server("open_graph-graph-rag")

# 延迟加载的全局实例
_retriever = None
_graph_manager = None
_graph_edit_service = None  # type: Optional[object]
_bridge_supervisor = None  # type: Optional[object]


def _get_data_dir() -> Path:
    import os
    data_dir = os.environ.get("open_graph_DATA_DIR")
    if data_dir:
        return Path(data_dir)
    return ROOT / "data"


def _get_retriever():
    global _retriever
    if _retriever is None:
        from src.graph.manager import LightweightGraphManager
        from src.retrieval.hybrid_retriever import GraphRAGRetriever
        db_path = _get_data_dir() / "kg_graph.db"
        if not db_path.exists():
            raise FileNotFoundError(f"图数据库不存在: {db_path}")
        gm = LightweightGraphManager(str(db_path))
        _retriever = GraphRAGRetriever(gm)
    return _retriever


def _get_graph_manager():
    global _graph_manager
    if _graph_manager is None:
        from src.graph.manager import LightweightGraphManager
        db_path = _get_data_dir() / "kg_graph.db"
        _graph_manager = LightweightGraphManager(str(db_path))
    return _graph_manager


def _get_graph_edit_service():
    global _graph_edit_service
    if _graph_edit_service is None:
        from src.server.graph_edit_service import GraphEditService

        def _reload():
            global _graph_manager, _retriever
            _graph_manager = None
            _retriever = None
            _get_graph_manager()

        _graph_edit_service = GraphEditService(
            graph_manager=_get_graph_manager(),
            data_dir=_get_data_dir(),
            reload_graph_state=_reload,
        )
    return _graph_edit_service


def _clamp_int(value: Any, default: int, min_value: int, max_value: int) -> int:
    try:
        iv = int(value)
    except Exception:
        iv = default
    return max(min_value, min(max_value, iv))


def _safe_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, default=str)


def _strip_embedding_in_nodes(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for node in nodes:
        item = dict(node)
        item.pop("_embedding", None)
        out.append(item)
    return out


def _extract_code_asset_link(gm, entry: dict[str, Any]) -> str:
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

    import os
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
    retriever = _get_retriever()
    top_k = _clamp_int(top_k, 5, 1, 100)
    depth = _clamp_int(depth, 1, 0, 2)

    if depth >= 1:
        result = retriever.deep_retrieve(query=query, top_k=top_k, mode=mode, label_filter=label_filter)
        if depth == 1:
            result["ppr_expanded"] = []
            result["paths"] = []
            result["narrative"] = retriever._build_narrative(result["seeds"], result["neighbors"], [], [], [])
        return result["narrative"]

    rows = retriever.retrieve(query=query, top_k=top_k, mode=mode, label_filter=label_filter)
    return _safe_json(rows)


async def _handle_query_default(args: dict):
    query = args["query"]
    top_k = args.get("top_k", 5)
    depth = args.get("depth", 1)
    label_filter = args.get("label_filter", "")
    text = _run_query_default(query=query, top_k=top_k, depth=depth, label_filter=label_filter, mode="hybrid")
    return [TextContent(type="text", text=text)]


async def _handle_query_code_semantic(args: dict):
    query = args["query"]
    top_k = _clamp_int(args.get("top_k", 5), 5, 1, 100)
    max_files = _clamp_int(args.get("max_files", 5), 5, 0, 50)
    include_repo_map = bool(args.get("include_repo_map", True))
    repo_map_timeout_ms = _clamp_int(args.get("repo_map_timeout_ms", 3500), 3500, 100, 15000)

    retriever = _get_retriever()
    gm = _get_graph_manager()

    rows = retriever.retrieve(query=query, top_k=top_k, mode="dense", label_filter="CodeFile")
    code_index = []
    for item in rows:
        code_index.append({
            "id": item.get("id"),
            "name": item.get("name"),
            "label": item.get("label"),
            "score": item.get("score", 0.0),
            "asset_link": _extract_code_asset_link(gm, item),
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

    payload = {
        "results": code_index,
        "code_index": code_index,
        "repo_map": repo_map,
        "repo_map_status": repo_map_status,
        "repo_map_meta": repo_map_meta,
        "result_type": "code_search",
    }
    return [TextContent(type="text", text=_safe_json(payload))]


async def _handle_query_custom(args: dict):
    query = args["query"]
    mode = args.get("mode", "hybrid")
    top_k = _clamp_int(args.get("top_k", 8), 8, 1, 100)
    label_filter = args.get("label_filter", "")
    depth = _clamp_int(args.get("depth", 2), 2, 0, 2)
    expand_node_ids = args.get("expand_node_ids", "[]")
    expand_hops = _clamp_int(args.get("expand_hops", 2), 2, 1, 3)
    expand_max_size = _clamp_int(args.get("expand_max_size", 200), 200, 1, 400)
    include_paths = bool(args.get("include_paths", True))
    include_schema = bool(args.get("include_schema", False))

    retriever = _get_retriever()
    gm = _get_graph_manager()

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

    try:
        parsed_expand_ids = json.loads(expand_node_ids) if expand_node_ids else []
    except json.JSONDecodeError:
        return [TextContent(type="text", text=_safe_json({"error": "expand_node_ids 不是合法的 JSON 数组"}))]

    if not isinstance(parsed_expand_ids, list):
        return [TextContent(type="text", text=_safe_json({"error": "expand_node_ids 必须是 JSON 数组"}))]

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

    payload = {
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
    }
    return [TextContent(type="text", text=_safe_json(payload))]


# ============================================================================
# MCP 工具注册
# ============================================================================

@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="query_default",
            description="默认查询入口：最少参数，默认 hybrid 检索并返回可读结果。",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "查询内容"},
                    "top_k": {"type": "integer", "description": "返回结果数量", "default": 5},
                    "depth": {"type": "integer", "description": "检索深度 0-2", "default": 1},
                    "label_filter": {"type": "string", "description": "标签过滤", "default": ""},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="query_code_semantic",
            description="代码语义查询入口：固定代码语义检索路径，可附加 Repo Map。",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "代码查询内容"},
                    "top_k": {"type": "integer", "description": "返回代码结果数量", "default": 5},
                    "max_files": {"type": "integer", "description": "最多生成 Repo Map 的文件数", "default": 5},
                    "include_repo_map": {"type": "boolean", "description": "是否附加 Repo Map", "default": True},
                    "repo_map_timeout_ms": {"type": "integer", "description": "Repo Map 超时毫秒", "default": 3500},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="query_custom",
            description="自定义查询入口：可控检索参数与图展开参数。",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "查询内容"},
                    "mode": {
                        "type": "string",
                        "description": "检索模式",
                        "default": "hybrid",
                        "enum": ["hybrid", "dense", "bm25", "code_semantic"],
                    },
                    "top_k": {"type": "integer", "description": "返回结果数量", "default": 8},
                    "label_filter": {"type": "string", "description": "标签过滤", "default": ""},
                    "depth": {"type": "integer", "description": "检索深度 0-2", "default": 2},
                    "expand_node_ids": {"type": "string", "description": "展开中心节点 JSON 数组字符串", "default": "[]"},
                    "expand_hops": {"type": "integer", "description": "展开跳数", "default": 2},
                    "expand_max_size": {"type": "integer", "description": "展开最大节点规模", "default": 200},
                    "include_paths": {"type": "boolean", "description": "是否包含路径分析", "default": True},
                    "include_schema": {"type": "boolean", "description": "是否附加 schema 概览", "default": False},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="kg_query",
            description=(
                "查询 open_graph 图知识库。支持 hybrid(Dense+BM25), dense(仅向量), bm25(仅关键词) 三种模式。"
                "返回相关业务术语及其描述、关联文档、代码路径。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "查询内容"},
                    "top_k": {"type": "integer", "description": "返回结果数量", "default": 5},
                    "mode": {
                        "type": "string",
                        "description": "检索模式: hybrid/dense/bm25/code_semantic/custom",
                        "default": "hybrid",
                        "enum": ["hybrid", "dense", "bm25", "code_semantic", "custom"]
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="term_lookup",
            description="精确查找指定业务术语的完整信息: 描述、别名、关联文档、代码文件、架构节点。",
            inputSchema={
                "type": "object",
                "properties": {
                    "term_name": {"type": "string", "description": "业务术语名称"},
                },
                "required": ["term_name"],
            },
        ),
        Tool(
            name="kg_stats",
            description="获取图知识库统计: 节点数/边数/标签分布/边类型分布。",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="kg_build",
            description=(
                "启动/恢复知识图谱构建流水线。"
                "长时间运行，会定期保存 checkpoint。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "phase_start": {
                        "type": "integer",
                        "description": "从哪个阶段开始 (0=全部, 1/2/3=指定phase)",
                        "default": 0,
                    },
                    "concurrency": {
                        "type": "integer",
                        "description": "并发数",
                        "default": 3,
                    },
                    "test_mode": {
                        "type": "boolean",
                        "description": "测试模式(仅3个术语)",
                        "default": False,
                    },
                },
            },
        ),
        Tool(
            name="kg_build_status",
            description="查看当前建库进度: 已完成术语数、各 phase 完成率。",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        # ── 新增: 图编辑与管理工具 ──
        Tool(
            name="kg_node_edit",
            description=(
                "图节点 CRUD 操作。"
                "action=get: 按 ID 获取节点。"
                "action=create: 创建节点(需 node_id, label, properties)。"
                "action=update: 部分更新节点属性(合并而非覆盖)。"
                "action=delete: 删除节点及其所有边。"
                "action=list_by_label: 按标签列出节点(支持 limit/offset/filter_key/filter_value)。"
                "设置 dry_run=true 可预览写操作而不实际执行。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "操作类型",
                        "enum": ["get", "create", "update", "delete", "list_by_label"],
                    },
                    "node_id": {"type": "string", "description": "节点 ID"},
                    "label": {"type": "string", "description": "节点标签 (create/list_by_label)"},
                    "properties": {
                        "type": "object",
                        "description": "节点属性 (create/update)",
                    },
                    "limit": {"type": "integer", "description": "list_by_label 返回数量上限", "default": 50},
                    "offset": {"type": "integer", "description": "list_by_label 跳过数量", "default": 0},
                    "filter_key": {"type": "string", "description": "list_by_label 属性过滤键"},
                    "filter_value": {"type": "string", "description": "list_by_label 属性过滤值(子串匹配)"},
                    "dry_run": {"type": "boolean", "description": "预览模式: 不实际执行写操作", "default": False},
                },
                "required": ["action"],
            },
        ),
        Tool(
            name="kg_edge_edit",
            description=(
                "图边 CRUD 操作。"
                "action=get: 获取两节点间的边。"
                "action=create: 创建/更新边。"
                "action=delete: 删除指定边。"
                "action=list: 列出节点的所有出边/入边。"
                "设置 dry_run=true 可预览写操作而不实际执行。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "操作类型",
                        "enum": ["get", "create", "delete", "list"],
                    },
                    "source": {"type": "string", "description": "源节点 ID"},
                    "target": {"type": "string", "description": "目标节点 ID"},
                    "relationship": {"type": "string", "description": "关系类型"},
                    "properties": {"type": "object", "description": "边属性 (create)"},
                    "node_id": {"type": "string", "description": "list 时的节点 ID"},
                    "direction": {
                        "type": "string",
                        "description": "list 方向: OUT/IN/BOTH",
                        "default": "BOTH",
                        "enum": ["OUT", "IN", "BOTH"],
                    },
                    "dry_run": {"type": "boolean", "description": "预览模式: 不实际执行写操作", "default": False},
                },
                "required": ["action"],
            },
        ),
        Tool(
            name="kg_merge_nodes",
            description=(
                "合并多个节点到一个目标节点: 所有入边/出边重定向到目标，源节点可选删除。"
                "典型用途: 合并重复的 BusinessTerm 或 InBusinessEntity。"
                "设置 dry_run=true 可预览合并影响而不实际执行。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "source_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "要合并的源节点 ID 列表",
                    },
                    "target_id": {"type": "string", "description": "合并目标节点 ID"},
                    "delete_sources": {
                        "type": "boolean",
                        "description": "是否删除源节点(false 则标记 merged_into)",
                        "default": False,
                    },
                    "dry_run": {"type": "boolean", "description": "预览模式: 不实际执行合并", "default": False},
                },
                "required": ["source_ids", "target_id"],
            },
        ),
        Tool(
            name="kg_tree_op",
            description=(
                "树结构操作。"
                "action=remove_and_reparent: 删除树节点，子节点自动连接到其父节点。"
                "action=insert_between: 在父子之间插入新节点(父->子 变为 父->新->子)。"
                "设置 dry_run=true 可预览操作而不实际执行。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["remove_and_reparent", "insert_between"],
                    },
                    "node_id": {"type": "string", "description": "要操作的节点 ID"},
                    "relationship": {"type": "string", "description": "树边关系类型(如 SUBFOLDER_OF, HAS_CHILD)"},
                    "new_node_id": {"type": "string", "description": "insert_between: 新节点 ID"},
                    "new_node_label": {"type": "string", "description": "insert_between: 新节点标签"},
                    "new_node_properties": {"type": "object", "description": "insert_between: 新节点属性"},
                    "parent_id": {"type": "string", "description": "insert_between: 父节点 ID"},
                    "child_id": {"type": "string", "description": "insert_between: 子节点 ID"},
                    "dry_run": {"type": "boolean", "description": "预览模式: 不实际执行操作", "default": False},
                },
                "required": ["action", "relationship"],
            },
        ),
        Tool(
            name="kg_neighbors",
            description=(
                "获取节点的 N 跳邻域(节点+边)。用于探索节点周围的图结构。"
                "默认跳过 L1 基础设施节点(如 BaseView、ResourceManager)以避免搜索爆炸。"
                "L1 节点会被聚合展示在 '依赖的基础设施' 区域。"
                "设置 skip_infra=false 可关闭此优化。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "node_id": {"type": "string", "description": "中心节点 ID"},
                    "max_hops": {"type": "integer", "description": "最大跳数", "default": 1},
                    "relationship": {"type": "string", "description": "只沿特定关系类型扩展(可选)"},
                    "skip_infra": {
                        "type": "boolean",
                        "description": "是否跳过 L1 基础设施节点(默认 true)",
                        "default": True,
                    },
                },
                "required": ["node_id"],
            },
        ),
        Tool(
            name="kg_version",
            description=(
                "数据库版本管理（快照系统）。"
                "action=list: 列出所有快照。"
                "action=search: 按关键词搜索快照(keyword)。"
                "action=save: 创建新快照(需 message)。回退前会自动保存。"
                "action=rollback: 回退到指定快照(需 snapshot_id)，回退前自动保存当前状态。"
                "action=delete: 删除指定快照。"
                "save/rollback/delete 支持 dry_run=true 预览。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "操作类型",
                        "enum": ["list", "search", "save", "rollback", "delete"],
                    },
                    "message": {"type": "string", "description": "save: 快照描述"},
                    "keyword": {"type": "string", "description": "search: 搜索关键词"},
                    "snapshot_id": {"type": "string", "description": "rollback/delete: 快照 ID (如 v001)"},
                    "dry_run": {"type": "boolean", "description": "预览模式: 不实际执行写操作", "default": False},
                },
                "required": ["action"],
            },
        ),
        Tool(
            name="kg_questions",
            description=(
                "Question 管理工具。"
                "action=list: 列出 pending 的 Question(支持 severity/category 过滤)。"
                "action=resolve: 标记为 resolved 并填写 answer。"
                "action=dismiss: 标记为 dismissed(误报)。"
                "action=stats: Question 统计(按 category/severity 分布)。"
                "设置 dry_run=true 可预览 resolve/dismiss 操作而不实际执行。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["list", "resolve", "dismiss", "stats"],
                    },
                    "question_id": {"type": "string", "description": "resolve/dismiss: Question 节点 ID"},
                    "answer": {"type": "string", "description": "resolve: 回答内容"},
                    "severity": {"type": "string", "description": "list 过滤: high/medium/low"},
                    "category": {"type": "string", "description": "list 过滤: weak_association/untraceable/contradictory/wrong_association/ambiguous"},
                    "limit": {"type": "integer", "description": "list 返回数量上限", "default": 20},
                    "dry_run": {"type": "boolean", "description": "预览模式: 不实际执行写操作", "default": False},
                },
                "required": ["action"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict):
    try:
        if name == "query_default":
            return await _handle_query_default(arguments)
        elif name == "query_code_semantic":
            return await _handle_query_code_semantic(arguments)
        elif name == "query_custom":
            return await _handle_query_custom(arguments)
        elif name == "kg_query":
            return await _handle_kg_query(arguments)
        elif name == "term_lookup":
            return await _handle_term_lookup(arguments)
        elif name == "kg_stats":
            return await _handle_kg_stats(arguments)
        elif name == "kg_build":
            return await _handle_kg_build(arguments)
        elif name == "kg_build_status":
            return await _handle_kg_build_status(arguments)
        elif name == "kg_node_edit":
            return await _handle_kg_node_edit(arguments)
        elif name == "kg_edge_edit":
            return await _handle_kg_edge_edit(arguments)
        elif name == "kg_merge_nodes":
            return await _handle_kg_merge_nodes(arguments)
        elif name == "kg_tree_op":
            return await _handle_kg_tree_op(arguments)
        elif name == "kg_neighbors":
            return await _handle_kg_neighbors(arguments)
        elif name == "kg_version":
            return await _handle_kg_version(arguments)
        elif name == "kg_questions":
            return await _handle_kg_questions(arguments)
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
    except Exception as e:
        logger.error(f"Tool {name} error: {e}", exc_info=True)
        return [TextContent(type="text", text=f"Error: {e}")]


# ============================================================================
# 工具处理函数
# ============================================================================

async def _handle_kg_query(args: dict):
    mode = args.get("mode", "hybrid")

    if mode == "code_semantic":
        forwarded = {
            "query": args["query"],
            "top_k": args.get("top_k", 5),
            "max_files": args.get("max_files", 5),
            "include_repo_map": args.get("include_repo_map", True),
            "repo_map_timeout_ms": args.get("repo_map_timeout_ms", 3500),
        }
        return await _handle_query_code_semantic(forwarded)

    if mode == "custom":
        forwarded = {
            "query": args["query"],
            "mode": "hybrid",
            "top_k": args.get("top_k", 8),
            "label_filter": args.get("label_filter", ""),
            "depth": args.get("depth", 2),
            "expand_node_ids": args.get("expand_node_ids", "[]"),
            "expand_hops": args.get("expand_hops", 2),
            "expand_max_size": args.get("expand_max_size", 200),
            "include_paths": args.get("include_paths", True),
            "include_schema": args.get("include_schema", False),
        }
        return await _handle_query_custom(forwarded)

    query = args["query"]
    top_k = args.get("top_k", 5)
    depth = args.get("depth", 1)
    label_filter = args.get("label_filter", "")

    text = _run_query_default(query=query, top_k=top_k, depth=depth, label_filter=label_filter, mode=mode)
    return [TextContent(type="text", text=text)]


async def _handle_term_lookup(args: dict):
    term_name = args["term_name"]
    gm = _get_graph_manager()

    # 尝试精确匹配
    nid = f"bt:{term_name}"
    node_data = gm.get_node(nid)

    if not node_data:
        # 尝试模糊匹配
        all_terms = gm.find_nodes_by_label("BusinessTerm")
        matches = [t for t in all_terms if term_name.lower() in t.get("name", "").lower()]
        if matches:
            lines = [f"未找到精确匹配 '{term_name}'，但找到以下相似术语:"]
            for t in matches[:10]:
                lines.append(f"  - {t.get('name', t['id'])}: {t.get('description', '')[:100]}")
            return [TextContent(type="text", text="\n".join(lines))]
        return [TextContent(type="text", text=f"未找到术语: {term_name}")]

    # 构建详细信息
    lines = [f"# {node_data.get('name', term_name)}"]
    if node_data.get("description"):
        lines.append(f"\n{node_data['description']}")

    if node_data.get("aliases"):
        lines.append(f"\n**别名:** {', '.join(node_data['aliases'])}")
    if node_data.get("dev_names"):
        lines.append(f"**开发名:** {', '.join(node_data['dev_names'])}")
    if node_data.get("dev_hint"):
        lines.append(f"**开发代号:** {node_data['dev_hint']}")
    if node_data.get("parent_category"):
        lines.append(f"**所属分类:** {node_data['parent_category']}")

    # 关联节点
    related = gm.find_related_nodes(nid, direction="BOTH")
    if related:
        lines.append("\n**关联节点:**")
        for r in related[:15]:
            edge = r.get("_edge", {})
            rel = edge.get("relationship", "?")
            lines.append(f"  - [{rel}] {r.get('name', r['id'])} ({r.get('label', '')})")

    return [TextContent(type="text", text="\n".join(lines))]


async def _handle_kg_stats(args: dict):
    gm = _get_graph_manager()

    total_nodes = gm.node_count()
    total_edges = gm.edge_count()
    schema = gm.get_schema_info("overview")

    lines = [
        f"# 图知识库统计",
        f"",
        f"**总节点数:** {total_nodes}",
        f"**总边数:** {total_edges}",
        f"",
        f"## 节点分布",
    ]
    for label, count in schema.get("node_labels", {}).items():
        lines.append(f"  - {label}: {count}")

    lines.append(f"\n## 边类型分布")
    for rel, count in schema.get("edge_types", {}).items():
        lines.append(f"  - {rel}: {count}")

    return [TextContent(type="text", text="\n".join(lines))]


async def _handle_kg_build(args: dict):
    phase_start = args.get("phase_start", 0)
    concurrency = args.get("concurrency", 3)
    test_mode = args.get("test_mode", False)

    from src.ingest.pipeline import run_local_test, main as pipeline_main

    if test_mode:
        ckpt, report = await run_local_test(phase_start=max(phase_start, 1), concurrency=concurrency)
        return [TextContent(type="text", text=f"测试构建完成!\n\n{json.dumps(report, ensure_ascii=False, indent=2)}")]

    # 完整构建 — 模拟命令行参数
    import sys
    old_argv = sys.argv
    sys.argv = ["pipeline"]
    if phase_start > 0:
        sys.argv.extend(["--phase", str(phase_start)])
    sys.argv.extend(["--concurrency", str(concurrency)])

    try:
        await pipeline_main()
        return [TextContent(type="text", text="图谱构建完成!")]
    finally:
        sys.argv = old_argv


async def _handle_kg_node_edit(args: dict):
    from src.server.graph_edit_service import format_edit_result_text
    svc = _get_graph_edit_service()
    result = svc.execute_node_edit(args)
    return [TextContent(type="text", text=format_edit_result_text(result))]


async def _handle_kg_edge_edit(args: dict):
    from src.server.graph_edit_service import format_edit_result_text
    svc = _get_graph_edit_service()
    result = svc.execute_edge_edit(args)
    return [TextContent(type="text", text=format_edit_result_text(result))]


async def _handle_kg_merge_nodes(args: dict):
    from src.server.graph_edit_service import format_edit_result_text
    svc = _get_graph_edit_service()
    result = svc.execute_merge_nodes(args)
    return [TextContent(type="text", text=format_edit_result_text(result))]


async def _handle_kg_tree_op(args: dict):
    from src.server.graph_edit_service import format_edit_result_text
    svc = _get_graph_edit_service()
    result = svc.execute_tree_op(args)
    return [TextContent(type="text", text=format_edit_result_text(result))]


async def _handle_kg_neighbors(args: dict):
    gm = _get_graph_manager()
    node_id = args.get("node_id", "")
    max_hops = args.get("max_hops", 1)
    rel = args.get("relationship")
    skip_infra = args.get("skip_infra", True)

    if not node_id:
        return [TextContent(type="text", text="需要 node_id")]

    result = gm.get_neighbors(node_id, max_hops, rel, skip_infra=skip_infra)
    nodes = result["nodes"]
    edges = result["edges"]
    infra_summary = result.get("infra_summary", [])

    lines = [f"节点 {node_id} 的 {max_hops} 跳邻域: {len(nodes)} 节点, {len(edges)} 边\n"]

    if infra_summary:
        lines.append("## 依赖的基础设施 (L1, 已聚合)")
        for inf in infra_summary[:15]:
            cls = inf.get("class_name", "")
            cls_str = f" ({cls})" if cls else ""
            lines.append(f"  {inf['name']}{cls_str} — 被引用 {inf['ref_count']} 次")
        if len(infra_summary) > 15:
            lines.append(f"  ... 及另外 {len(infra_summary) - 15} 个基础设施节点")
        lines.append("")

    lines.append("## 节点")
    for n in nodes[:60]:
        nid = n.get("id", "?")
        label = n.get("label", "?")
        name = n.get("name", "")
        lines.append(f"  [{label}] {nid}" + (f": {name}" if name else ""))
    if len(nodes) > 60:
        lines.append(f"  ... 及另外 {len(nodes) - 60} 个")

    lines.append("\n## 边")
    for e in edges[:80]:
        lines.append(f"  {e['source'][:50]} -[{e['relationship']}]-> {e['target'][:50]}")
    if len(edges) > 80:
        lines.append(f"  ... 及另外 {len(edges) - 80} 条")

    return [TextContent(type="text", text="\n".join(lines))]


async def _handle_kg_questions(args: dict):
    from src.server.graph_edit_service import format_edit_result_text
    svc = _get_graph_edit_service()
    result = svc.execute_questions(args)
    return [TextContent(type="text", text=format_edit_result_text(result))]


async def _handle_kg_version(args: dict):
    """数据库版本管理 (git 快照系统)"""
    from src.server.graph_edit_service import format_edit_result_text
    svc = _get_graph_edit_service()
    result = svc.execute_version(args)
    return [TextContent(type="text", text=format_edit_result_text(result))]


async def _handle_kg_build_status(args: dict):
    data_dir = _get_data_dir()
    ckpt_path = data_dir / "knowledge_graph" / "checkpoint.json"

    if not ckpt_path.exists():
        return [TextContent(type="text", text="尚未开始构建（无 checkpoint 文件）。")]

    with open(ckpt_path, "r", encoding="utf-8") as f:
        ckpt_data = json.load(f)

    terms = ckpt_data.get("terms", {})
    total = len(terms)

    phase_stats = {"phase1": 0, "phase2": 0, "phase3": 0}
    errors = {"phase1": 0, "phase2": 0, "phase3": 0}

    for name, td in terms.items():
        for phase in ["phase1", "phase2", "phase3"]:
            pd = td.get(phase)
            if pd is not None:
                if isinstance(pd, dict) and pd.get("_status") == "error":
                    errors[phase] += 1
                else:
                    phase_stats[phase] += 1

    meta = ckpt_data.get("metadata", {})
    last_saved = meta.get("last_saved", "未知")

    lines = [
        f"# 建库进度",
        f"",
        f"**总术语数:** {total}",
        f"**最后更新:** {last_saved}",
        f"",
        f"| Phase | 完成 | 错误 | 进度 |",
        f"|-------|------|------|------|",
    ]
    for phase in ["phase1", "phase2", "phase3"]:
        done = phase_stats[phase]
        err = errors[phase]
        pct = f"{done/total*100:.1f}%" if total > 0 else "0%"
        lines.append(f"| {phase} | {done} | {err} | {pct} |")

    return [TextContent(type="text", text="\n".join(lines))]


# ============================================================================
# 入口
# ============================================================================

def _init_shared():
    """初始化共享的图数据库、检索器、问题管理器实例"""
    global _retriever, _graph_manager, _graph_edit_service
    logger.info("初始化共享实例...")

    from src.graph.manager import LightweightGraphManager
    from src.retrieval.hybrid_retriever import GraphRAGRetriever
    from src.tools.question_manager import QuestionManager

    db_path = _get_data_dir() / "kg_graph.db"
    if not db_path.exists():
        raise FileNotFoundError(f"图数据库不存在: {db_path}")

    gm = LightweightGraphManager(str(db_path))
    ret = GraphRAGRetriever(gm)
    qm = QuestionManager(gm)

    # 设置本模块的全局变量（stdio MCP 工具使用）
    _graph_manager = gm
    _retriever = ret

    # 初始化统一编辑服务（stdio + HTTP 共享）
    _graph_edit_service = None  # 清除旧实例，让 _get_graph_edit_service() 重新创建
    _get_graph_edit_service()

    logger.info("共享实例初始化完成: gm=%d nodes, retriever ready", gm.node_count())
    return gm, ret, qm


async def _start_web_server(gm, qm, host="0.0.0.0", port=8000):
    """后台启动 Web UI + API 服务 (:8000)"""
    try:
        import sys
        import uvicorn
        from src.server.main import app, init_app

        # 在后台模式下避免 uvicorn 访问已关闭的 stdout
        if getattr(sys.stdout, "closed", False):
            sys.stdout = sys.stderr

        init_app(gm, qm)
        config = uvicorn.Config(
            app,
            host=host,
            port=port,
            log_level="warning",
            log_config=None,
            access_log=False,
            use_colors=False,
        )
        srv = uvicorn.Server(config)
        logger.info(f"启动 Web 服务: http://{host}:{port}")
        await srv.serve()
    except Exception:
        logger.exception("Web 服务启动失败")


async def _start_mcp_http(gm, ret, qm, host="0.0.0.0", port=8001):
    """后台启动 MCP HTTP 服务 (:8001)"""
    try:
        import sys
        from src.mcp_server import init_shared, run_async

        # 在后台模式下避免 uvicorn 访问已关闭的 stdout
        if getattr(sys.stdout, "closed", False):
            sys.stdout = sys.stderr

        init_shared(gm, ret, qm)
        await run_async(host=host, port=port)
    except Exception:
        logger.exception("MCP HTTP 服务启动失败")


def _ensure_embeddings():
    """检查并补齐缺失的 embedding，启动时自动运行"""
    try:
        from src.ingest.generate_embeddings import generate_embeddings, DB_PATH
        logger.info("[Embedding] 检查缺失的 embedding...")
        # 在后台运行时禁用 tqdm 进度条（避免与 stdio 冲突）
        import os
        old_environ = os.environ.get('TQDM_DISABLE')
        os.environ['TQDM_DISABLE'] = '1'
        try:
            generate_embeddings(db_path=DB_PATH, force=False)
        finally:
            if old_environ is None:
                os.environ.pop('TQDM_DISABLE', None)
            else:
                os.environ['TQDM_DISABLE'] = old_environ
        logger.info("[Embedding] 补齐完成")
    except Exception:
        logger.exception("[Embedding] 自动补齐失败（不影响服务运行）")


def _get_bridge_supervisor():
    """延迟初始化桥接守护器。"""
    global _bridge_supervisor
    if _bridge_supervisor is None:
        from src.server.bridge_supervisor import BridgeSupervisor
        _bridge_supervisor = BridgeSupervisor()
    return _bridge_supervisor


async def _start_bridge_supervisor():
    """启动桥接守护任务（若已配置）。"""
    try:
        sup = _get_bridge_supervisor()
        await sup.start()
    except Exception:
        logger.exception("[bridge_supervisor] 启动失败")


async def _stop_bridge_supervisor():
    """停止桥接守护任务。"""
    global _bridge_supervisor
    if _bridge_supervisor is None:
        return
    try:
        await _bridge_supervisor.stop()
    except Exception:
        logger.exception("[bridge_supervisor] 停止失败")


async def _deferred_init():
    """后台初始化共享实例 + HTTP 服务，不阻塞 stdio MCP 握手"""
    import os

    # 启动桥接守护（尽早执行，避免首个请求时桥接未就绪）
    await _start_bridge_supervisor()

    # 重量级初始化（加载 2.3GB 图数据库 + embedding 模型，约 30s）
    gm, ret, qm = await asyncio.get_event_loop().run_in_executor(None, _init_shared)

    # 自动补齐缺失的 embedding（在线程池中运行，不阻塞服务启动）
    asyncio.ensure_future(
        asyncio.get_event_loop().run_in_executor(None, _ensure_embeddings)
    )

    # 启动 HTTP 服务
    web_port = int(os.environ.get("open_graph_WEB_PORT", "8000"))
    mcp_port = int(os.environ.get("open_graph_MCP_PORT", "8001"))
    asyncio.ensure_future(_start_web_server(gm, qm, port=web_port))
    asyncio.ensure_future(_start_mcp_http(gm, ret, qm, port=mcp_port))


async def run():
    """以 stdio 模式运行 MCP 服务器，同时后台启动 HTTP 服务"""
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    logger.info("Starting open_graph GraphRAG MCP Server (unified)...")

    # 1. 后台初始化（不阻塞 stdio MCP 握手）
    init_task = asyncio.ensure_future(_deferred_init())

    # 2. stdio MCP 主循环 — 立即启动，工具首次调用时会等待 lazy init
    #    如果 stdin 是 tty/pipe（gateway 连接），正常运行 stdio MCP
    #    如果 stdin 是 /dev/null（独立运行），跳过 stdio，只跑 HTTP 服务
    import os
    if os.isatty(0) or not sys.stdin.closed:
        try:
            async with stdio_server() as (read_stream, write_stream):
                await server.run(read_stream, write_stream, server.create_initialization_options())
        except Exception:
            logger.info("stdio MCP 已断开，HTTP 服务继续运行")

    # stdio 结束后，等待后台初始化和 HTTP 服务
    await init_task

    # stdio 退出后停止桥接守护
    await _stop_bridge_supervisor()


def main():
    """
    统一入口，根据环境变量决定运行模式

    模式说明:
    - standalone: 同时运行 stdio MCP + HTTP MCP + Web UI (默认)
    - http: 仅运行 HTTP MCP + Web UI (无 stdio)
    - stdio: 仅运行 stdio MCP (向后兼容 open_graph_agent 插件模式)
    """
    import os

    mode = os.environ.get("open_graph_MODE", "standalone")
    host = os.environ.get("open_graph_HOST", "0.0.0.0")
    web_port = int(os.environ.get("open_graph_WEB_PORT", "8000"))
    mcp_port = int(os.environ.get("open_graph_MCP_PORT", "8001"))

    logging.basicConfig(
        level=os.environ.get("open_graph_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.info(f"open_graph Graph RAG 启动中 [mode={mode}]...")

    if mode == "stdio":
        # 纯 stdio 模式：被 open_graph_agent 通过 pipe 连接
        logger.info("运行模式：stdio MCP only")
        asyncio.run(run())
    elif mode == "http":
        # 纯 HTTP 模式：无 stdio，仅 HTTP MCP + Web
        logger.info(f"运行模式：HTTP MCP + Web (host={host}, web_port={web_port}, mcp_port={mcp_port})")
        asyncio.run(run_http_only(host=host, web_port=web_port, mcp_port=mcp_port))
    else:
        # standalone 模式：stdio + HTTP + Web
        logger.info(f"运行模式：standalone (stdio + HTTP + Web)")
        logger.info(f"  Web UI: http://{host}:{web_port}")
        logger.info(f"  MCP HTTP: http://{host}:{mcp_port}/mcp")
        logger.info(f"  Health: http://{host}:{mcp_port}/health")
        asyncio.run(run())


async def run_http_only(host: str = "0.0.0.0", web_port: int = 8000, mcp_port: int = 8001):
    """
    仅运行 HTTP MCP + Web UI 模式（无 stdio）
    适用于独立部署场景

    注意：embedding 生成会在后台异步运行，不影响服务启动
    """
    logger.info("正在初始化图数据库和检索器...")

    # 启动桥接守护（尽早执行）
    await _start_bridge_supervisor()

    # 同步初始化
    gm, ret, qm = _init_shared()

    logger.info("启动 Web 服务和 MCP HTTP 服务...")

    # 先启动 Web 和 MCP HTTP 服务（立即响应请求）
    web_task = asyncio.create_task(_start_web_server(gm, qm, host=host, port=web_port))
    mcp_task = asyncio.create_task(_start_mcp_http(gm, ret, qm, host=host, port=mcp_port))

    # 等待服务启动
    await asyncio.sleep(2)

    # 在后台异步生成 embedding（不阻塞服务）
    asyncio.get_event_loop().run_in_executor(None, _ensure_embeddings)

    try:
        # 等待两个服务任务完成
        await asyncio.gather(web_task, mcp_task)
    finally:
        await _stop_bridge_supervisor()


def run_stdio_mode():
    """入口函数：纯 stdio 模式"""
    asyncio.run(run())


def run_http_mode():
    """入口函数：纯 HTTP 模式"""
    import os
    host = os.environ.get("open_graph_HOST", "0.0.0.0")
    web_port = int(os.environ.get("open_graph_WEB_PORT", "8000"))
    mcp_port = int(os.environ.get("open_graph_MCP_PORT", "8001"))
    asyncio.run(run_http_only(host=host, web_port=web_port, mcp_port=mcp_port))


if __name__ == "__main__":
    main()
