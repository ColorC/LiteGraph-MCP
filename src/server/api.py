# -*- coding: utf-8 -*-
"""
API Routes for Knowledge Graph and Question Management
"""

import logging
from typing import List, Optional, Any, Dict, Set
from fastapi import APIRouter, HTTPException, Query, Body
from pydantic import BaseModel, Field

from src.tools.question_manager import QuestionManager

logger = logging.getLogger(__name__)

router = APIRouter()

# Dependency injection for QuestionManager should be handled in main.py or via a dependency provider
# For now, we assume a global or injected instance will be available.
# This might need adjustment based on how the server is structured.
# Let's assume we can get it from app.state or similar, but for simplicity here I'll define a way to set it.

_question_manager: Optional[QuestionManager] = None
_proposal_consumer = None  # ProposalConsumer 实例引用
_edit_service = None  # GraphEditService 实例引用

def set_question_manager(qm: QuestionManager):
    global _question_manager
    _question_manager = qm

def set_proposal_consumer(consumer):
    global _proposal_consumer
    _proposal_consumer = consumer

def set_edit_service(svc):
    global _edit_service
    _edit_service = svc

def get_question_manager() -> QuestionManager:
    if _question_manager is None:
        raise HTTPException(status_code=500, detail="QuestionManager not initialized")
    return _question_manager

def _get_edit_service():
    global _edit_service
    if _edit_service is not None:
        return _edit_service
    # Lazy init from question manager's graph manager
    try:
        qm = get_question_manager()
        from src.server.graph_edit_service import GraphEditService
        import os
        from pathlib import Path
        data_dir = os.environ.get("open_graph_DATA_DIR") or str(Path(__file__).resolve().parents[2] / "data")
        _edit_service = GraphEditService(
            graph_manager=qm.gm,
            data_dir=Path(data_dir),
        )
        return _edit_service
    except Exception:
        raise HTTPException(status_code=500, detail="GraphEditService not initialized")

# --- Models ---

class QuestionCreateRequest(BaseModel):
    question: str
    category: str
    context: str
    related_node_id: str
    extra_props: Optional[Dict[str, Any]] = None

class QuestionResponse(BaseModel):
    id: str
    question: str
    category: str
    status: str
    context: str
    related_node_id: str
    answer: Optional[str] = None
    created_at: str
    # Add other fields as needed from existing graph props

class ApproveRequest(BaseModel):
    refined_answer: Optional[str] = None

class RejectRequest(BaseModel):
    reason: str

# --- Helpers ---

# 需要从API响应中剥离的内部字段
_STRIP_KEYS = {"_embedding"}

def _strip_internal(obj):
    """从节点/问题字典中移除内部字段（如 _embedding）"""
    if isinstance(obj, dict):
        return {k: v for k, v in obj.items() if k not in _STRIP_KEYS}
    return obj

# --- Routes ---

@router.get("/questions", response_model=Dict[str, Any])
async def list_questions(
    category: Optional[str] = Query(None, description="Filter by category"),
    status: Optional[str] = Query("pending", description="Filter by status"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Page size"),
    keyword: Optional[str] = Query(None, description="Search keyword")
):
    """
    List questions with pagination.
    """
    qm = get_question_manager()
    skip = (page - 1) * page_size

    if status == "pending":
         questions, total = qm.list_pending(category=category, skip=skip, limit=page_size, keyword=keyword)
    else:
         questions = qm.list_all(status=status, category=category)
         if keyword:
             k_lower = keyword.lower()
             questions = [
                 q for q in questions
                 if k_lower in q.get("question", "").lower() or k_lower in q.get("context", "").lower()
             ]
         total = len(questions)
         questions = questions[skip : skip + page_size]
    
    return {
        "items": [_strip_internal(q) for q in questions],
        "total": total,
        "page": page,
        "page_size": page_size
    }


@router.post("/questions", response_model=Dict[str, str])
async def create_question(request: QuestionCreateRequest):
    """
    Create a new question.
    """
    qm = get_question_manager()
    try:
        req_dict = request.model_dump()
        qid = qm.create_question(
            question=req_dict["question"],
            category=req_dict["category"],
            context=req_dict["context"],
            related_node_id=req_dict["related_node_id"],
            extra_props=req_dict.get("extra_props")
        )
        return {"id": qid}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/questions/{question_id}/approve")
async def approve_question(question_id: str, request: Optional[ApproveRequest] = Body(None)):
    """
    Approve a question (sets status to 'approved').
    If refined_answer is provided, it's saved as the answer without changing status to 'answered'.
    """
    qm = get_question_manager()
    success = qm.approve_question(question_id)

    # 如果有自定义审批意见，直接写入 answer 字段（不改变 approved 状态）
    if success and request and request.refined_answer:
        node_data = qm.gm.get_node(question_id)
        if node_data:
            node_data["answer"] = request.refined_answer
            node_data["auto_pass_no_review"] = False
            label = node_data.pop("label", "Question")
            node_data.pop("id", None)
            qm.gm.add_node(question_id, label, node_data)
    elif success:
        # 无审批意见时，标记为可直接通过，不走 LLM
        node_data = qm.gm.get_node(question_id)
        if node_data:
            node_data["auto_pass_no_review"] = True
            label = node_data.pop("label", "Question")
            node_data.pop("id", None)
            qm.gm.add_node(question_id, label, node_data)
         
    if not success:
        raise HTTPException(status_code=404, detail="Question not found or operation failed")
    return {"status": "success", "id": question_id}

@router.post("/questions/{question_id}/reject")
async def reject_question(question_id: str, request: RejectRequest = Body(...)):
    """
    Reject a question (sets status to 'rejected').
    Reason is required.
    """
    qm = get_question_manager()
    success = qm.reject_question(question_id, reason=request.reason)
    if not success:
        raise HTTPException(status_code=404, detail="Question not found or operation failed")
    return {"status": "success", "id": question_id}

# --- Graph Visualization Routes ---

@router.get("/graph/nodes/{node_id}", response_model=Dict[str, Any])
async def get_node(node_id: str):
    """
    Get node details by ID.
    """
    qm = get_question_manager()
    # Decode double encoded ID if needed? FastAPI handles path param unquoting usually.
    # But if frontend uses encodeURIComponent, then {node_id} will be 'entity%3AScheduleType_2'.
    # Starlette/FastAPI automatically decodes ONE level.
    # So if frontend sends /api/graph/nodes/entity%3AScheduleType_2, node_id will be "entity:ScheduleType_2".
    
    if qm.gm.has_node(node_id):
        node_data = qm.gm.get_node(node_id)
        return _strip_internal(node_data)
    else:
        # Try finding case-insensitive?
        # Or maybe it's a "partial" ID? User input might be vague, but here it comes from related_node_id which should be exact.
        raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found")

@router.get("/graph/search", response_model=List[Dict[str, Any]])
async def search_nodes(
    q: str = Query(..., description="Search query"),
    limit: int = 20
):
    """
    Search for nodes in the graph.
    """
    qm = get_question_manager()
    results = qm.gm.search_nodes(q, limit)
    return [_strip_internal(r) for r in results]

@router.get("/graph/neighbors", response_model=List[Dict[str, Any]])
async def get_neighbors(
    node_id: str = Query(..., description="Node ID"),
    direction: str = Query("BOTH", description="Direction: OUT, IN, BOTH")
):
    """
    Get neighbors of a node.
    """
    qm = get_question_manager()
    # Note: find_related_nodes returns dicts with 'id' and properties
    return [_strip_internal(n) for n in qm.gm.find_related_nodes(node_id, direction=direction)]

@router.get("/graph/labels", response_model=List[str])
async def get_graph_labels():
    """
    Get all node labels.
    """
    qm = get_question_manager()
    return qm.gm.get_all_labels()

@router.get("/graph/node-types", response_model=List[Dict[str, Any]])
async def get_node_types():
    """
    Get all node types with their counts.
    """
    qm = get_question_manager()
    return qm.gm.get_node_types_with_counts()

@router.get("/graph/sample", response_model=Dict[str, Any])
async def get_sample_graph(
    limit: int = 50,
    strategy: str = "mesh",
    labels: Optional[str] = Query(None, description="Comma-separated labels to filter by")
):
    """
    Get a sample of nodes.
    strategy='mesh' returns a connected subgraph (snowball).
    strategy='random' returns random nodes.
    labels: comma-separated label filter (e.g. 'CodeFile,Story')
    """
    qm = get_question_manager()
    label_list = [l.strip() for l in labels.split(",") if l.strip()] if labels else None

    if strategy == "random":
        nodes_data = qm.gm.random_nodes(limit, labels=label_list)
        node_ids = {n["id"] for n in nodes_data}
        links_data = []
        for n in nodes_data:
            edges = qm.gm.find_edges(n["id"], direction="OUT")
            for e in edges:
                if e["target"] in node_ids:
                    links_data.append(_strip_internal(e))
        return {"nodes": [_strip_internal(n) for n in nodes_data], "links": links_data}
    else:
        result = qm.gm.get_random_subgraph(limit, labels=label_list)
        return {
            "nodes": [_strip_internal(n) for n in result["nodes"]],
            "links": [_strip_internal(e) for e in result["edges"]]
        }

@router.post("/graph/apply", response_model=Dict[str, Any])
async def apply_graph_change(
    action: str = Body(..., description="Action type: add_edge, enrich_logic"),
    params: Dict[str, Any] = Body(..., description="Action parameters")
):
    """
    Apply a verification change to the graph (compatibility endpoint).
    Intended for Approval Queue execution.
    New integrations should use /graph/node/edit, /graph/edge/edit, etc.
    """
    svc = _get_edit_service()
    result = svc.execute_apply(action, params)
    if not result.get("ok"):
        err = result.get("error", {})
        raise HTTPException(status_code=err.get("status", 400), detail=err.get("message", "Unknown error"))
    return result

@router.get("/graph/stats")
async def get_graph_stats():
    """
    Get graph statistics.
    """
    qm = get_question_manager()
    return {
        "nodes": qm.gm.node_count(),
        "edges": qm.gm.edge_count(),
        "questions": qm.get_stats()
    }

# -------------------------------------------------------------------------
# Proposals 监控 API
# -------------------------------------------------------------------------

# Proposal 相关的状态（经过审批流的，不含普通 pending）
_PROPOSAL_STATUSES = {"approved", "in_progress", "completed", "completed_no_action", "failed", "rejected"}
# 人工提议的 pending 也显示在执行监控中
_PROPOSAL_CATEGORIES = {"proposal"}

def _is_proposal_visible(q: dict) -> bool:
    """判断一个 Question 是否应该在执行监控中显示"""
    s = q.get("status", "pending")
    if s in _PROPOSAL_STATUSES:
        return True
    # 人工提议的 pending 也显示
    if s == "pending" and q.get("category") in _PROPOSAL_CATEGORIES:
        return True
    return False

@router.get("/proposals/stats")
async def get_proposal_stats():
    """获取 proposal 处理统计"""
    qm = get_question_manager()
    all_q = qm.gm.find_nodes_by_label("Question")

    stats = {s: 0 for s in _PROPOSAL_STATUSES}
    stats["pending_proposal"] = 0
    stats["total"] = 0
    for q in all_q:
        if not _is_proposal_visible(q):
            continue
        s = q.get("status", "pending")
        if s == "pending":
            stats["pending_proposal"] += 1
        else:
            stats[s] = stats.get(s, 0) + 1
        stats["total"] += 1

    # 附加 consumer 运行时统计
    consumer_stats = {}
    if _proposal_consumer:
        consumer_stats = _proposal_consumer.get_stats()
    stats["consumer"] = consumer_stats

    return stats

@router.get("/proposals/history")
async def get_proposal_history(
    status: Optional[str] = Query(None, description="Filter by status"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    keyword: Optional[str] = Query(None, description="Search keyword"),
):
    """获取 proposal 处理历史（分页）"""
    qm = get_question_manager()
    all_q = qm.gm.find_nodes_by_label("Question")

    # 过滤
    results = []
    for q in all_q:
        if not _is_proposal_visible(q):
            continue
        q_status = q.get("status", "pending")
        if status and status != "all" and q_status != status:
            continue
        if keyword:
            kw = keyword.lower()
            if kw not in q.get("question", "").lower() and kw not in q.get("context", "").lower():
                continue
        results.append(_strip_internal(q))

    # 按时间倒序（优先显示最近的）
    results.sort(key=lambda x: x.get("completed_at") or x.get("approved_at") or x.get("created_at") or "", reverse=True)

    total = len(results)
    skip = (page - 1) * page_size
    items = results[skip:skip + page_size]

    return {"items": items, "total": total, "page": page, "page_size": page_size}


# -------------------------------------------------------------------------
# 查询实验 API
# -------------------------------------------------------------------------

class QueryLabSearchRequest(BaseModel):
    query: str
    mode: str = "hybrid"
    top_k: int = 10
    label_filter: str = ""
    depth: int = 0

    # index_only + RepoMap advanced options (backward compatible)
    repo_map_semantic_rerank: bool = False
    repo_map_score_threshold: float = 0.0
    repo_map_alpha: float = 0.7
    repo_map_max_files: int = 20
    repo_map_max_symbols_per_file: int = 30
    repo_map_order: str = "top_down"  # top_down | score_only
    repo_map_include_signature_details: bool = True
    repo_map_timeout_ms: int = 3500

    # Adaptive guard: keep index_only not worse than hybrid+CodeFile baseline
    repo_map_strategy: str = "adaptive_guard"  # adaptive_guard | baseline_hybrid | advanced_rerank
    repo_map_direct_confidence_threshold: float = 0.35
    repo_map_hybrid_guard_weight: float = 0.65
    repo_map_enable_term_community_fallback: bool = True
    repo_map_protect_baseline_top_n: int = 8

    # custom expand mode (agent)
    expand_node_ids: List[str] = Field(default_factory=list)
    expand_top_n: int = 3
    expand_hops: int = 2
    expand_max_size: int = 180

    # payload: 默认返回紧凑可读结构，原始大图需显式开启
    include_raw_graph: bool = False

class QueryLabSqlRequest(BaseModel):
    sql: str
    db: str = "main"

class QueryLabNeighborsRequest(BaseModel):
    node_id: str
    hops: int = 1

class QueryLabPathRequest(BaseModel):
    source_id: str
    target_id: str
    max_depth: int = 6


def _normalize_query_lab_mode(raw_mode: str) -> str:
    mode = (raw_mode or "").strip().lower()
    if mode in {"", "default", "hybrid", "dense"}:
        return "default"
    if mode in {"code_semantic", "index_only"}:
        return "code_semantic"
    if mode in {"custom", "bm25", "exact"}:
        return "custom"
    return "default"


@router.post("/query-lab/search")
async def query_lab_search(req: QueryLabSearchRequest):
    """查询实验 - 统一检索入口"""
    import time
    from src.retrieval.hybrid_retriever import get_graph_retriever

    qm = get_question_manager()
    retriever = get_graph_retriever()

    t0 = time.time()
    requested_mode = req.mode
    effective_mode = _normalize_query_lab_mode(requested_mode)
    try:
        if effective_mode == "custom" and (requested_mode or "").strip().lower() == "exact":
            node = qm.gm.get_node(req.query)
            results = [_strip_internal(node)] if node else []
        elif effective_mode == "custom":
            excluded_labels = {"Question"}
            centers: List[Dict[str, Any]] = []
            seen_center_ids: Set[str] = set()

            for node_id in (req.expand_node_ids or []):
                nid = str(node_id).strip()
                if not nid or nid in seen_center_ids:
                    continue
                node_data = qm.gm.get_node(nid)
                if not node_data:
                    continue
                label = str(node_data.get("label", ""))
                if label in excluded_labels:
                    continue
                centers.append({
                    "id": nid,
                    "label": label,
                    "name": node_data.get("name", nid),
                })
                seen_center_ids.add(nid)

            if not centers:
                seed_hits = retriever.retrieve(
                    query=req.query,
                    top_k=max(req.top_k * 4, 40),
                    mode="hybrid",
                    label_filter=req.label_filter,
                )
                for hit in seed_hits:
                    nid = str(hit.get("id", "")).strip()
                    if not nid or nid in seen_center_ids:
                        continue
                    label = str(hit.get("label", ""))
                    if label in excluded_labels:
                        continue
                    centers.append({
                        "id": nid,
                        "label": label,
                        "name": hit.get("name", nid),
                        "score": float(hit.get("score", 0.0)),
                    })
                    seen_center_ids.add(nid)
                    if len(centers) >= max(1, int(req.expand_top_n or 3)):
                        break

            expand_hops = max(1, min(4, int(req.expand_hops or 2)))
            expand_max_size = max(40, min(800, int(req.expand_max_size or 180)))

            graph_nodes: Dict[str, Dict[str, Any]] = {}
            graph_edges: Dict[tuple, Dict[str, Any]] = {}
            center_ids = {str(c.get("id", "")) for c in centers}

            for center in centers:
                center_id = str(center.get("id", ""))
                if not center_id:
                    continue
                g = qm.gm.get_neighbors(center_id, max_hops=expand_hops, max_size=expand_max_size)
                for node in (g or {}).get("nodes", []):
                    node_id = str(node.get("id", ""))
                    if not node_id:
                        continue
                    node_label = str(node.get("label", ""))
                    if node_label in excluded_labels:
                        continue
                    graph_nodes[node_id] = {
                        "id": node_id,
                        "label": node_label,
                        "name": node.get("name", node_id),
                        "description": str(node.get("description", ""))[:180],
                        "is_center": node_id in center_ids,
                    }
                for edge in (g or {}).get("edges", []):
                    src = str(edge.get("source", ""))
                    dst = str(edge.get("target", ""))
                    if not src or not dst:
                        continue
                    if src not in graph_nodes or dst not in graph_nodes:
                        continue
                    edge_key = (src, dst, str(edge.get("type", "related")))
                    graph_edges[edge_key] = {
                        "source": src,
                        "target": dst,
                        "relationship": str(edge.get("type", "related")),
                    }

            expanded_graph = {
                "nodes": list(graph_nodes.values()),
                "edges": list(graph_edges.values()),
                "centers": centers,
                "hops": expand_hops,
            }

            elapsed_ms = int((time.time() - t0) * 1000)
            return {
                "results": expanded_graph,
                "expanded_graph": expanded_graph,
                "expansion_meta": {
                    "requested_expand_node_ids": [str(x) for x in (req.expand_node_ids or []) if str(x).strip()],
                    "selected_centers": len(centers),
                    "expand_top_n": max(1, int(req.expand_top_n or 3)),
                    "expand_hops": expand_hops,
                    "expand_max_size": expand_max_size,
                },
                "elapsed_ms": elapsed_ms,
                "mode": requested_mode,
                "requested_mode": requested_mode,
                "effective_mode": effective_mode,
                "query": req.query,
                "result_type": "graph",
                "nodes": expanded_graph["nodes"],
                "edges": expanded_graph["edges"],
            }
        elif effective_mode == "code_semantic":
            # Index-Only：代码优先，术语聚合作为辅助信息
            import math
            import os
            import re
            from src.tools.repo_map import RepoMapTool

            def _to_record_path(node_id: str, node_data: dict) -> str:
                path = str(node_data.get("path", "")).replace("\\", "/")
                if path:
                    for prefix in ("C:/Git/record/", "C:\\Git\\record\\", "c:/git/record/", "c:\\git\\record\\"):
                        if path.lower().startswith(prefix.lower()):
                            path = path[len(prefix):]
                            break
                    return f"record/{path}".replace("//", "/")
                if node_id.startswith("file:"):
                    return f"record/{node_id[5:]}".replace("//", "/")
                return ""

            def _safe_norm_score(raw: float, max_raw: float) -> float:
                if max_raw <= 0:
                    return 0.0
                return max(0.0, min(1.0, raw / max_raw))

            def _tokenize(text: str) -> List[str]:
                return [t for t in re.split(r"[^\w\u4e00-\u9fff]+", (text or "").lower()) if t]

            def _lexical_overlap(query_text: str, target_text: str) -> float:
                q_tokens = _tokenize(query_text)
                if not q_tokens:
                    return 0.0
                target = (target_text or "").lower()
                hits = sum(1 for t in q_tokens if t in target)
                return hits / max(1, len(q_tokens))

            def _expand_query_terms(query_text: str, term_candidates: List[Dict[str, Any]]) -> List[str]:
                """针对业务高频词做意图展开，并吸收术语候选别名。"""
                expanded = set(_tokenize(query_text))
                q_lower = (query_text or "").lower()

                # 面向当前核心案例（抽卡/招募）的轻量意图词扩展
                intent_lexicon = {
                    "抽卡": ["招募", "卡池", "up招募", "英雄抽卡", "tavern", "神秘屋", "stargazer", "gacha"],
                    "招募": ["抽卡", "卡池", "up招募", "tavern", "神秘屋", "英雄抽卡", "stargazer", "gacha"],
                    "卡池": ["抽卡", "招募", "up招募", "tavern", "神秘屋", "stargazer", "gacha"],
                }

                for intent, aliases in intent_lexicon.items():
                    if intent in q_lower:
                        for alias in aliases:
                            expanded.update(_tokenize(alias))

                # 融合检索到的业务术语及别名（避免完全依赖静态词典）
                for hit in term_candidates[:8]:
                    name = str(hit.get("name", ""))
                    if name:
                        expanded.update(_tokenize(name))

                    node_id = str(hit.get("id", ""))
                    if not node_id:
                        continue
                    node_data = qm.gm.get_node(node_id) or {}
                    aliases = node_data.get("discovered_aliases") or node_data.get("aliases") or []
                    for alias in aliases[:8]:
                        expanded.update(_tokenize(str(alias)))

                return [t for t in expanded if t]

            def _token_hit_ratio(tokens: List[str], target_text: str) -> float:
                if not tokens:
                    return 0.0
                target = (target_text or "").lower()
                hits = sum(1 for t in tokens if t and t in target)
                return hits / max(1, len(tokens))

            def _is_dev_english_alias(alias_text: str) -> bool:
                a = (alias_text or "").strip()
                if not a:
                    return False
                if len(a) < 2 or len(a) > 64:
                    return False
                has_ascii_alpha = bool(re.search(r"[A-Za-z]", a))
                code_like = bool(re.fullmatch(r"[A-Za-z][A-Za-z0-9_./:-]*", a))
                return has_ascii_alpha and code_like

            def _prioritize_aliases(raw_aliases: List[Any], limit: int = 3) -> List[str]:
                normalized: List[str] = []
                seen = set()
                for raw in raw_aliases or []:
                    text = str(raw).strip()
                    if not text:
                        continue
                    key = text.lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    normalized.append(text)

                if not normalized:
                    return []

                dev_aliases = [a for a in normalized if _is_dev_english_alias(a)]
                other_aliases = [a for a in normalized if a not in dev_aliases]
                return (dev_aliases + other_aliases)[:max(1, limit)]

            requested_repo_map_max_files = max(1, int(req.repo_map_max_files or req.top_k or 1))
            runtime_repo_map_max_files = (
                max(1, min(requested_repo_map_max_files, 8))
                if effective_mode == "code_semantic"
                else max(1, min(requested_repo_map_max_files, 40))
            )

            # 1) 代码检索（主）：直接用 hybrid 索引，按 CodeFile 过滤
            code_hits = retriever.retrieve(
                query=req.query,
                # 重要：label_filter 在 retriever 内部是“先融合排序再过滤”，
                # 若 top_k 过小会把大量 CodeFile 提前截断掉（例如 Tavern.lua 被挤出）。
                # 这里强制更大 over-fetch，后续再由本层 final_score + max_files 控制输出。
                top_k=max(runtime_repo_map_max_files * 12, req.top_k * 10, 120),
                mode="hybrid",
                label_filter="CodeFile",
            )

            # 2) 术语检索（辅）：聚合 BusinessTerm + InBusinessEntity，并按重要性降噪
            raw_term_hits = []
            for term_label in ("BusinessTerm", "InBusinessEntity"):
                raw_term_hits.extend(
                    retriever.retrieve(
                        query=req.query,
                        top_k=max(req.top_k * 2, 12),
                        mode="hybrid",
                        label_filter=term_label,
                    )
                )

            def _rank_term_hits(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
                dedup_terms: Dict[str, Dict[str, Any]] = {}
                for hit in candidates:
                    node_id = str(hit.get("id", ""))
                    if not node_id:
                        continue
                    if node_id not in dedup_terms or float(hit.get("score", 0.0)) > float(dedup_terms[node_id].get("score", 0.0)):
                        dedup_terms[node_id] = hit

                if not dedup_terms:
                    return []

                max_raw = max(float(h.get("score", 0.0)) for h in dedup_terms.values())
                ranked: List[Dict[str, Any]] = []
                for hit in dedup_terms.values():
                    node_id = str(hit.get("id", ""))
                    node_data = qm.gm.get_node(node_id) or {}
                    label = str(hit.get("label") or node_data.get("label") or "")
                    if label not in {"BusinessTerm", "InBusinessEntity"}:
                        continue

                    degree = len(qm.gm.find_edges(node_id, direction="BOTH") or [])
                    name = str(hit.get("name", node_id))
                    aliases = node_data.get("discovered_aliases") or node_data.get("aliases") or []
                    prioritized_aliases = _prioritize_aliases(aliases, limit=5)
                    alias_text = " ".join(prioritized_aliases)
                    relevance = _lexical_overlap(req.query, f"{name} {alias_text}")
                    raw_norm = _safe_norm_score(float(hit.get("score", 0.0)), max_raw)
                    degree_norm = min(1.0, degree / 20.0)
                    label_bonus = 0.10 if label == "BusinessTerm" else 0.0
                    importance = min(1.0, 0.45 * raw_norm + 0.30 * relevance + 0.25 * degree_norm + label_bonus)

                    # InBusinessEntity 仅保留连接更丰富的项，减少术语噪声
                    if label == "InBusinessEntity" and degree < 3:
                        continue
                    if importance < 0.25:
                        continue

                    hit = dict(hit)
                    hit["importance"] = importance
                    hit["degree"] = degree
                    ranked.append(hit)

                ranked.sort(
                    key=lambda x: (
                        -float(x.get("importance", 0.0)),
                        -int(x.get("degree", 0)),
                        -float(x.get("score", 0.0)),
                        str(x.get("name", "")),
                    )
                )
                return ranked

            term_hits = _rank_term_hits(raw_term_hits)

            def _build_term_community_code_support(limit_terms: int = 6, max_size: int = 260) -> Dict[str, float]:
                support: Dict[str, float] = {}
                community_hops = 2 if effective_mode == "code_semantic" else 1
                for term in term_hits[:max(1, limit_terms)]:
                    term_id = str(term.get("id", ""))
                    if not term_id:
                        continue
                    term_importance = max(0.0, float(term.get("importance", 0.0)))
                    if term_importance <= 0.0:
                        term_importance = 0.1
                    nbr = qm.gm.get_neighbors(term_id, max_hops=community_hops, max_size=max_size)
                    for node in (nbr or {}).get("nodes", []):
                        if node.get("label") != "CodeFile":
                            continue
                        node_id = str(node.get("id", ""))
                        if not node_id:
                            continue
                        support[node_id] = support.get(node_id, 0.0) + term_importance
                return support

            term_community_code_support = _build_term_community_code_support()
            max_term_community_support = max(list(term_community_code_support.values()) + [0.0])

            def _collect_term_neighbor_code_hits(limit_terms: int = 3, max_size: int = 200) -> List[Dict[str, Any]]:
                neighbor_code_hits = []
                seen_code_ids = set()
                for term in term_hits[:max(1, limit_terms)]:
                    nbr = qm.gm.get_neighbors(term["id"], max_hops=1, max_size=max_size)
                    for node in (nbr or {}).get("nodes", []):
                        if node.get("label") != "CodeFile":
                            continue
                        nid = str(node.get("id", ""))
                        if not nid or nid in seen_code_ids:
                            continue
                        seen_code_ids.add(nid)
                        neighbor_code_hits.append({
                            "id": nid,
                            "name": node.get("name", nid),
                            "label": "CodeFile",
                            "score": 0.0,
                            "match_reason": "term_community_fallback",
                        })
                return neighbor_code_hits

            # 若代码直接命中为空，尝试从 top 术语的 1-hop 邻域补充代码
            if not code_hits and term_hits:
                code_hits = _collect_term_neighbor_code_hits()

            # 仅允许这些代码后缀参与 RepoMap（过滤 json/xml 等非代码资产）
            valid_repo_map_exts = {".py", ".cs", ".lua"}
            valid_file_kinds = {"script", "code", "source", "lua", "python", "csharp", "cs"}

            def _is_repo_map_target(path_str: str) -> bool:
                lower = path_str.lower()
                return any(lower.endswith(ext) for ext in valid_repo_map_exts)

            def _is_codefile_candidate(node_data: Dict[str, Any], node_id: str, target_path: str) -> bool:
                if _is_repo_map_target(target_path):
                    return True
                file_type = str(node_data.get("file_type", "")).lower()
                if file_type in valid_file_kinds:
                    return True
                lower_id = (node_id or "").lower()
                lower_path = (target_path or "").lower()
                return any(lower_id.endswith(ext) or lower_path.endswith(ext) for ext in valid_repo_map_exts)

            # 收集代码候选
            filtered_items = []
            for hit in code_hits:
                node_id = str(hit.get("id", ""))
                if not node_id:
                    continue
                node_data = qm.gm.get_node(node_id) or {}
                record_path = _to_record_path(node_id, node_data)
                target_path = record_path or node_id
                if not _is_codefile_candidate(node_data, node_id, target_path):
                    continue
                filtered_items.append({
                    "hit": hit,
                    "node_id": node_id,
                    "node_data": node_data,
                    "record_path": record_path,
                    "name": str(hit.get("name", node_id)),
                    "raw_score": float(hit.get("score", 0.0)),
                })

            # 去重（按 record_path / node_id）
            dedup = {}
            for item in filtered_items:
                key = item["record_path"] or item["node_id"]
                if key not in dedup or item["raw_score"] > dedup[key]["raw_score"]:
                    dedup[key] = item
            filtered_items = list(dedup.values())
            baseline_items = sorted(filtered_items, key=lambda x: (-(x["raw_score"]), x["name"]))

            def _merge_code_hits(existing_items: List[Dict[str, Any]], extra_hits: List[Dict[str, Any]], score_scale: float = 1.0) -> int:
                if not extra_hits:
                    return 0
                seen = {(it["record_path"] or it["node_id"]) for it in existing_items}
                added = 0
                for hit in extra_hits:
                    node_id = str(hit.get("id", ""))
                    if not node_id:
                        continue
                    node_data = qm.gm.get_node(node_id) or {}
                    record_path = _to_record_path(node_id, node_data)
                    target_path = record_path or node_id
                    if not _is_codefile_candidate(node_data, node_id, target_path):
                        continue
                    key = record_path or node_id
                    if key in seen:
                        continue
                    seen.add(key)
                    existing_items.append({
                        "hit": hit,
                        "node_id": node_id,
                        "node_data": node_data,
                        "record_path": record_path,
                        "name": str(hit.get("name", node_id)),
                        "raw_score": float(hit.get("score", 0.0)) * max(0.0, score_scale),
                    })
                    added += 1
                return added

            def _pick_expansion_queries(tokens: List[str], max_terms: int = 6) -> List[str]:
                if not tokens:
                    return []
                priority = ["tavern", "招募", "卡池", "stargazer", "gacha", "抽卡"]
                picked: List[str] = []
                seen = set()
                for p in priority:
                    if p in tokens and p not in seen:
                        picked.append(p)
                        seen.add(p)
                for t in tokens:
                    if t in seen:
                        continue
                    if len(t) < 2:
                        continue
                    picked.append(t)
                    seen.add(t)
                    if len(picked) >= max_terms:
                        break
                return picked[:max_terms]

            strategy = (req.repo_map_strategy or "adaptive_guard").strip().lower()
            if strategy not in {"adaptive_guard", "baseline_hybrid", "advanced_rerank"}:
                strategy = "adaptive_guard"
            strategy_auto_downgraded = False

            direct_conf_threshold = max(0.0, min(1.0, float(req.repo_map_direct_confidence_threshold)))
            hybrid_guard_weight = max(0.0, min(1.0, float(req.repo_map_hybrid_guard_weight)))
            enable_term_community_fallback = bool(req.repo_map_enable_term_community_fallback)
            protect_baseline_top_n = max(0, int(req.repo_map_protect_baseline_top_n or 0))

            semantic_mode_requested = bool(req.repo_map_semantic_rerank)
            semantic_mode_enabled = bool(
                semantic_mode_requested
                or effective_mode == "code_semantic"
                or (
                    effective_mode != "code_semantic"
                    and strategy in ("adaptive_guard", "advanced_rerank")
                )
            )
            semantic_scores: Dict[str, float] = {}
            bm25_scores: Dict[str, float] = {}
            scoring_mode = "lexical_only"
            used_term_community_fallback = False

            # code_semantic 默认更偏向语义与社群，降低纯关键词干扰
            semantic_alpha = max(0.0, min(1.0, float(req.repo_map_alpha)))
            if effective_mode == "code_semantic":
                semantic_alpha = max(semantic_alpha, 0.82)
            effective_hybrid_guard_weight = max(0.0, min(1.0, float(req.repo_map_hybrid_guard_weight)))
            if effective_mode == "code_semantic":
                effective_hybrid_guard_weight = min(effective_hybrid_guard_weight, 0.35)
            effective_protect_baseline_top_n = protect_baseline_top_n
            if effective_mode == "code_semantic":
                effective_protect_baseline_top_n = min(effective_protect_baseline_top_n, 1)

            expanded_query_tokens = _expand_query_terms(req.query, term_hits)
            query_intent_boost_enabled = bool(expanded_query_tokens)

            # Adaptive: 当直接命中置信度不足时，优先做意图词扩展检索；仍不足再启用术语社群补召回
            probe_conf = 0.0
            expansion_added = 0
            expansion_queries: List[str] = []
            added_neighbors = 0
            if strategy == "adaptive_guard" and term_hits:
                raw_max_probe = max([item["raw_score"] for item in filtered_items] + [0.0])
                probe_conf = 0.0
                for item in filtered_items:
                    node_id = item["node_id"]
                    path_text = item["record_path"] or node_id
                    lex_probe = _lexical_overlap(req.query, f"{item['name']} {path_text}")
                    raw_probe = _safe_norm_score(item["raw_score"], raw_max_probe)
                    probe_conf = max(probe_conf, 0.6 * lex_probe + 0.4 * raw_probe)

                expansion_added = 0
                expansion_queries = []
                if probe_conf < direct_conf_threshold:
                    expansion_queries = _pick_expansion_queries(expanded_query_tokens, max_terms=6)
                    for qx in expansion_queries:
                        if qx == (req.query or "").strip().lower():
                            continue
                        q_hits = retriever.retrieve(
                            query=qx,
                            top_k=max(req.repo_map_max_files * 3, 30),
                            mode="hybrid",
                            label_filter="CodeFile",
                        )
                        expansion_added += _merge_code_hits(filtered_items, q_hits, score_scale=0.92)

                if probe_conf < direct_conf_threshold and enable_term_community_fallback:
                    augmented_hits = _collect_term_neighbor_code_hits(limit_terms=4, max_size=260)
                    added_neighbors = _merge_code_hits(filtered_items, augmented_hits, score_scale=0.85)
                    used_term_community_fallback = bool(added_neighbors)
                else:
                    added_neighbors = 0

            # 可选：语义分（dense）+ 词法分（bm25）融合
            if semantic_mode_enabled:
                try:
                    dense_hits = retriever.retrieve(
                        query=req.query,
                        top_k=max(req.repo_map_max_files * 4, 40),
                        mode="dense",
                        label_filter="CodeFile",
                    )
                    sparse_hits = retriever.retrieve(
                        query=req.query,
                        top_k=max(req.repo_map_max_files * 4, 40),
                        mode="bm25",
                        label_filter="CodeFile",
                    )

                    dense_max = max([float(h.get("score", 0.0)) for h in dense_hits] + [0.0])
                    sparse_max = max([float(h.get("score", 0.0)) for h in sparse_hits] + [0.0])

                    semantic_scores = {
                        str(h.get("id", "")): _safe_norm_score(float(h.get("score", 0.0)), dense_max)
                        for h in dense_hits
                    }
                    bm25_scores = {
                        str(h.get("id", "")): _safe_norm_score(float(h.get("score", 0.0)), sparse_max)
                        for h in sparse_hits
                    }
                    scoring_mode = "hybrid_semantic_lexical"
                except Exception as rerank_err:
                    logger.warning(f"[index_only] 语义重排降级为 lexical-only: {rerank_err}")
                    semantic_mode_enabled = False
                    scoring_mode = "lexical_only"

            # 文件级打分
            hybrid_max = max([item["raw_score"] for item in filtered_items] + [0.0])
            alpha = semantic_alpha
            scored_items = []
            for item in filtered_items:
                node_id = item["node_id"]
                path_text = item["record_path"] or node_id
                baseline_hybrid_score = _safe_norm_score(item["raw_score"], hybrid_max)
                lexical_overlap = _lexical_overlap(req.query, f"{item['name']} {path_text}")
                lexical_sparse = bm25_scores.get(node_id, baseline_hybrid_score)
                lexical_score = 0.45 * lexical_overlap + 0.55 * lexical_sparse
                semantic_score = semantic_scores.get(node_id, baseline_hybrid_score)

                # 查询意图展开命中（术语中心先验）：提升“抽卡/招募”等高泛化词的核心文件排序
                intent_match = _token_hit_ratio(expanded_query_tokens, f"{item['name']} {path_text}")
                intent_boost = 0.18 * intent_match
                term_community_support_raw = term_community_code_support.get(node_id, 0.0)
                term_community_support = _safe_norm_score(term_community_support_raw, max_term_community_support)
                term_community_boost = 0.55 * term_community_support

                if semantic_mode_enabled:
                    base_score = alpha * semantic_score + (1.0 - alpha) * lexical_score
                else:
                    base_score = lexical_score

                advanced_score = min(1.0, base_score + intent_boost + term_community_boost)
                guard_score = effective_hybrid_guard_weight * baseline_hybrid_score + (1.0 - effective_hybrid_guard_weight) * advanced_score

                if strategy == "baseline_hybrid":
                    final_score = baseline_hybrid_score
                elif strategy == "advanced_rerank":
                    final_score = advanced_score
                else:
                    final_score = max(advanced_score, guard_score)

                if strategy == "adaptive_guard" and effective_protect_baseline_top_n > 0:
                    top_baseline_keys = {
                        (it["record_path"] or it["node_id"])
                        for it in baseline_items[:effective_protect_baseline_top_n]
                    }
                    this_key = item["record_path"] or item["node_id"]
                    if this_key in top_baseline_keys:
                        final_score = max(final_score, baseline_hybrid_score)

                item.update({
                    "baseline_hybrid_score": baseline_hybrid_score,
                    "lexical_score": lexical_score,
                    "semantic_score": semantic_score,
                    "intent_match": intent_match,
                    "intent_boost": intent_boost,
                    "term_community_support": term_community_support,
                    "term_community_boost": term_community_boost,
                    "advanced_score": advanced_score,
                    "guard_score": guard_score,
                    "final_score": final_score,
                })
                scored_items.append(item)

            scored_items.sort(key=lambda x: (-x["final_score"], -x.get("baseline_hybrid_score", 0.0), -(x["raw_score"]), x["name"]))

            if strategy == "adaptive_guard" and effective_protect_baseline_top_n > 0:
                protected_keys_in_order = [
                    (it["record_path"] or it["node_id"])
                    for it in baseline_items[:effective_protect_baseline_top_n]
                ]
                scored_by_key = {(it["record_path"] or it["node_id"]): it for it in scored_items}
                protected_items = [scored_by_key[k] for k in protected_keys_in_order if k in scored_by_key]
                protected_key_set = set(protected_keys_in_order)
                remaining_items = [
                    it for it in scored_items
                    if (it["record_path"] or it["node_id"]) not in protected_key_set
                ]
                scored_items = protected_items + remaining_items

            threshold = max(0.0, float(req.repo_map_score_threshold))
            after_threshold = [it for it in scored_items if it["final_score"] >= threshold]

            requested_max_files = int(req.repo_map_max_files or req.top_k or 1)
            if effective_mode == "code_semantic":
                max_files = max(1, min(requested_max_files, 8))
            else:
                max_files = max(1, min(requested_max_files, 30))
            selected_items = after_threshold[:max_files]

            def _confidence_bucket(score: float) -> str:
                if score >= 0.75:
                    return "high"
                if score >= 0.45:
                    return "medium"
                return "low"

            code_index: List[Dict[str, Any]] = []
            for idx, item in enumerate(selected_items, 1):
                score = float(item.get("final_score", 0.0))
                conf = _confidence_bucket(score)
                code_index.append({
                    "rank": idx,
                    "id": item["node_id"],
                    "name": item["name"],
                    "path": item["record_path"] or item["node_id"],
                    "confidence": conf,
                    "score": score,
                    "match_reason": "semantic_community_hybrid",
                })

            related_terms = []
            if term_hits:
                for hit in term_hits[: min(6, req.top_k)]:
                    node_id = str(hit.get("id", ""))
                    node_data = qm.gm.get_node(node_id) or {}
                    aliases = node_data.get("discovered_aliases") or node_data.get("aliases") or []
                    alias_items = _prioritize_aliases(aliases, limit=3)
                    related_terms.append({
                        "id": node_id,
                        "name": hit.get("name", node_id),
                        "importance": float(hit.get("importance", 0.0)),
                        "degree": int(hit.get("degree", 0)),
                        "aliases": alias_items,
                    })

            repo_map_meta = {
                "scoring_mode": scoring_mode,
                "strategy": strategy,
                "strategy_auto_downgraded": strategy_auto_downgraded,
                "requested_repo_map_max_files": requested_repo_map_max_files,
                "runtime_repo_map_max_files": runtime_repo_map_max_files,
                "semantic_rerank_requested": semantic_mode_requested,
                "semantic_rerank_enabled": semantic_mode_enabled,
                "direct_confidence_threshold": direct_conf_threshold,
                "hybrid_guard_weight": hybrid_guard_weight,
                "effective_hybrid_guard_weight": effective_hybrid_guard_weight,
                "term_community_fallback_enabled": enable_term_community_fallback,
                "term_community_fallback_used": used_term_community_fallback,
                "protect_baseline_top_n": protect_baseline_top_n,
                "effective_protect_baseline_top_n": effective_protect_baseline_top_n,
                "adaptive_probe_confidence": probe_conf if strategy == "adaptive_guard" else None,
                "alpha": alpha,
                "adaptive_expansion_queries": expansion_queries if strategy == "adaptive_guard" else [],
                "adaptive_expansion_added": expansion_added if strategy == "adaptive_guard" else 0,
                "adaptive_neighbor_added": added_neighbors if strategy == "adaptive_guard" else 0,
                "score_threshold": threshold,
                "max_files": max_files,
                "max_symbols_per_file": req.repo_map_max_symbols_per_file,
                "order": req.repo_map_order,
                "include_signature_details": req.repo_map_include_signature_details,
                "candidate_files": len(filtered_items),
                "selected_files": len(selected_items),
                "query_intent_boost_enabled": query_intent_boost_enabled,
                "expanded_query_tokens": expanded_query_tokens[:24],
                "semantic_priority_mode": effective_mode == "code_semantic",
            }

            repo_map = []
            repo_map_status = "ok"
            repo_map_limit = min(3, len(selected_items))
            bridge_url = os.environ.get('WINDOWS_FILE_BRIDGE_URL', '').strip()
            timeout_ms = max(800, min(10000, int(req.repo_map_timeout_ms or 3500)))
            repo_map_meta["repo_map_timeout_ms"] = timeout_ms

            if selected_items and not bridge_url:
                repo_map_status = "bridge_unavailable"
                repo_map_meta["generated_repo_maps"] = 0
                repo_map_meta["repo_map_generation_errors"] = ["WINDOWS_FILE_BRIDGE_URL is not configured"]
            elif selected_items:
                logger.info(f"[index_only] bridge_url: {bridge_url}")
                repo_tool = RepoMapTool(
                    root_path=os.getcwd(),
                    remote_bridge_url=bridge_url
                )

                generation_errors = []
                for item in selected_items[:repo_map_limit]:
                    file_path = item["record_path"] or item["node_id"]
                    logger.info(f"[index_only] 尝试生成 RepoMap: {file_path}")
                    try:
                        sig = repo_tool.generator.generate_for_file(file_path)
                        if not sig:
                            continue
                        rmap = sig.format(
                            include_signature_details=bool(req.repo_map_include_signature_details),
                            max_symbols_per_file=max(1, int(req.repo_map_max_symbols_per_file)),
                            query=req.query,
                            order=req.repo_map_order if req.repo_map_order in ("top_down", "score_only") else "top_down",
                        )
                        if rmap and "无法生成" not in rmap and "不存在" not in rmap:
                            repo_map.append({
                                "path": file_path,
                                "content": rmap,
                            })
                    except TimeoutError as e:
                        generation_errors.append(f"timeout:{file_path}:{e}")
                        repo_map_status = "timeout"
                    except Exception as e:
                        generation_errors.append(f"failed:{file_path}:{e}")
                        logger.warning(f"[index_only] RepoMap 生成失败 {file_path}: {e}")

                repo_map_meta["repo_map_generation_errors"] = generation_errors
                repo_map_meta["generated_repo_maps"] = len(repo_map)
                if repo_map_status != "timeout" and len(repo_map) == 0:
                    repo_map_status = "generation_failed"
            else:
                repo_map_status = "generation_failed"
                repo_map_meta["generated_repo_maps"] = 0

            elapsed_ms = int((time.time() - t0) * 1000)
            return {
                "results": code_index,
                "code_index": code_index,
                "repo_map": repo_map,
                "repo_map_status": repo_map_status,
                "related_terms": related_terms,
                "elapsed_ms": elapsed_ms,
                "mode": requested_mode,
                "requested_mode": requested_mode,
                "effective_mode": effective_mode,
                "query": req.query,
                "result_type": "code_search",
                "repo_map_meta": repo_map_meta,
            }
        elif effective_mode == "default":
            def _norm_local(raw: float, max_raw: float) -> float:
                if max_raw <= 0:
                    return 0.0
                return max(0.0, min(1.0, raw / max_raw))

            overfetch_k = max(req.top_k * 8, 80)
            focus_k = max(req.top_k * 4, 40)

            # 1) 全量搜索（先搜）
            full_hits = retriever.retrieve(
                query=req.query,
                top_k=overfetch_k,
                mode="hybrid",
                label_filter=""
            )

            # 2) 定向搜索（CodeFile + BusinessTerm only）用于重排增强
            code_hits = retriever.retrieve(
                query=req.query,
                top_k=focus_k,
                mode="hybrid",
                label_filter="CodeFile"
            )
            term_hits = retriever.retrieve(
                query=req.query,
                top_k=focus_k,
                mode="hybrid",
                label_filter="BusinessTerm"
            )

            focus_code_ids = {str(h.get("id", "")) for h in code_hits if h.get("id")}
            focus_term_ids = {str(h.get("id", "")) for h in term_hits if h.get("id")}

            # 术语支持度：由相关业务实体群体命中进行支撑（防止单个实体噪音）
            full_hit_score_map = {
                str(h.get("id", "")): float(h.get("score", 0.0))
                for h in full_hits
                if h.get("id")
            }
            term_entity_support: Dict[str, float] = {}
            for term in term_hits[: min(16, len(term_hits))]:
                term_id = str(term.get("id", ""))
                if not term_id:
                    continue
                neighbor_graph = qm.gm.get_neighbors(term_id, max_hops=1, max_size=260)
                support = 0.0
                for node in (neighbor_graph or {}).get("nodes", []):
                    if node.get("label") != "InBusinessEntity":
                        continue
                    entity_id = str(node.get("id", ""))
                    support += float(full_hit_score_map.get(entity_id, 0.0))
                term_entity_support[term_id] = support

            max_raw = max([float(h.get("score", 0.0)) for h in full_hits] + [0.0])
            max_term_support = max(list(term_entity_support.values()) + [0.0])

            # 先聚合再过滤，避免“先过滤再搜索”导致结果数量不可控
            excluded_core_labels = {"InBusinessEntity", "Question"}
            dedup_candidates: Dict[str, Dict[str, Any]] = {}
            for hit in full_hits + code_hits + term_hits:
                node_id = str(hit.get("id", ""))
                if not node_id:
                    continue
                prev = dedup_candidates.get(node_id)
                if not prev or float(hit.get("score", 0.0)) > float(prev.get("score", 0.0)):
                    dedup_candidates[node_id] = hit

            reranked = []
            for node_id, hit in dedup_candidates.items():
                label = str(hit.get("label", ""))

                # 默认搜索永远不把业务实体/内部问题节点作为核心结果
                if label in excluded_core_labels:
                    continue

                base_score = _norm_local(float(hit.get("score", 0.0)), max_raw)
                code_boost = 0.22 if node_id in focus_code_ids else 0.0
                term_boost = 0.30 if node_id in focus_term_ids else 0.0

                term_support = _norm_local(term_entity_support.get(node_id, 0.0), max_term_support)
                support_boost = 0.0
                support_penalty = 0.0
                if label == "BusinessTerm":
                    support_boost = 0.28 * term_support
                    if term_support < 0.12:
                        support_penalty = 0.14

                final_score = max(0.0, min(1.0, 0.60 * base_score + code_boost + term_boost + support_boost - support_penalty))

                reranked.append({
                    "id": node_id,
                    "score": final_score,
                    "raw_score": float(hit.get("score", 0.0)),
                    "label": label,
                    "name": hit.get("name", node_id),
                    "description": hit.get("description", ""),
                    "term_support": term_support if label == "BusinessTerm" else 0.0,
                })

            reranked.sort(key=lambda x: (-float(x.get("score", 0.0)), -float(x.get("raw_score", 0.0)), str(x.get("name", ""))))

            # 保留请求过滤能力，但 InBusinessEntity/Question 过滤请求在 default 下忽略
            requested_label_filter = (req.label_filter or "").strip()
            ignored_label_filter = False
            if requested_label_filter:
                if requested_label_filter in excluded_core_labels:
                    ignored_label_filter = True
                else:
                    reranked = [r for r in reranked if r.get("label") == requested_label_filter]

            max_results = max(1, req.top_k)
            max_term_results = max(2, max_results // 2)
            term_part = [r for r in reranked if r.get("label") == "BusinessTerm"][:max_term_results]
            term_ids = {str(r.get("id", "")) for r in term_part}
            other_part = [r for r in reranked if str(r.get("id", "")) not in term_ids]
            ordered_results = (term_part + other_part)[:max_results]

            # 主结果突出“业务相关快照可读性”，尾部结果仅保留索引字段
            core_labels = {"BusinessTerm", "WikiStory", "CodeFile"}
            core_target = max(4, min(max_results, 8))
            top_score = float(ordered_results[0].get("score", 0.0)) if ordered_results else 0.0
            core_score_threshold = max(0.28, top_score * 0.55)

            core_results = []
            tail_index = []
            for rank, item in enumerate(ordered_results, 1):
                enriched = dict(item)
                enriched["rank"] = rank
                is_core = (
                    rank <= core_target
                    or (
                        str(item.get("label", "")) in core_labels
                        and float(item.get("score", 0.0)) >= core_score_threshold
                    )
                )
                if is_core:
                    core_results.append(enriched)
                else:
                    tail_index.append({
                        "id": str(item.get("id", "")),
                        "rank": rank,
                        "label": str(item.get("label", "")),
                        "name": item.get("name", ""),
                        "score": float(item.get("score", 0.0)),
                        "index_only": True,
                    })

            results = core_results + tail_index

            # 3) 按重排序选择聚合中心并扩展多跳内容快照（保连通，优先可读资料）
            snapshot_hops = max(1, min(3, req.depth if req.depth > 0 else 2))
            snapshot_centers: List[Dict[str, Any]] = []
            seen_center_ids: Set[str] = set()
            preferred_center_labels = {"BusinessTerm", "WikiStory", "CodeFile", "InBusinessEntity"}

            for item in reranked:
                center_id = str(item.get("id", ""))
                center_label = str(item.get("label", ""))
                if not center_id or center_id in seen_center_ids:
                    continue
                if center_label not in preferred_center_labels:
                    continue
                snapshot_centers.append(item)
                seen_center_ids.add(center_id)
                if len(snapshot_centers) >= 3:
                    break

            if len(snapshot_centers) < 3:
                for item in core_results:
                    center_id = str(item.get("id", ""))
                    if not center_id or center_id in seen_center_ids:
                        continue
                    snapshot_centers.append(item)
                    seen_center_ids.add(center_id)
                    if len(snapshot_centers) >= 3:
                        break

            if len(snapshot_centers) < 3:
                for item in reranked:
                    center_id = str(item.get("id", ""))
                    if not center_id or center_id in seen_center_ids:
                        continue
                    snapshot_centers.append(item)
                    seen_center_ids.add(center_id)
                    if len(snapshot_centers) >= 3:
                        break

            term_snapshots = []
            snapshot_excluded_labels = {"Question"}
            preferred_snapshot_labels = {"WikiStory", "CodeFile"}
            for center in snapshot_centers:
                center_id = str(center.get("id", ""))
                if not center_id:
                    continue
                graph = qm.gm.get_neighbors(center_id, max_hops=snapshot_hops, max_size=180)
                node_map: Dict[str, Dict[str, Any]] = {}
                for node in (graph or {}).get("nodes", []):
                    node_label = str(node.get("label", ""))
                    if node_label in snapshot_excluded_labels:
                        continue
                    node_id = str(node.get("id", ""))
                    if not node_id:
                        continue
                    priority = 2 if node_label in preferred_snapshot_labels else (1 if node_label == "BusinessTerm" else 0)
                    node_map[node_id] = {
                        "id": node_id,
                        "label": node_label,
                        "name": node.get("name", node_id),
                        "description": str(node.get("description", ""))[:180],
                        "_priority": priority,
                    }

                filtered_edges = []
                connected_ids = set()
                for edge in (graph or {}).get("edges", []):
                    src = str(edge.get("source", ""))
                    dst = str(edge.get("target", ""))
                    if src in node_map and dst in node_map:
                        filtered_edges.append(_strip_internal(edge))
                        connected_ids.add(src)
                        connected_ids.add(dst)

                term_id = str(center.get("id", ""))
                selected_ids: List[str] = []
                if term_id in node_map:
                    selected_ids.append(term_id)

                def _rank_node_id(node_id: str):
                    n = node_map[node_id]
                    return (
                        -int(n.get("_priority", 0)),
                        str(n.get("label", "")) != "BusinessTerm",
                        str(n.get("name", "")),
                    )

                connected_preferred = [
                    nid for nid in connected_ids
                    if nid not in selected_ids and int(node_map[nid].get("_priority", 0)) >= 1
                ]
                connected_others = [
                    nid for nid in connected_ids
                    if nid not in selected_ids and nid not in connected_preferred
                ]
                isolated_preferred = [
                    nid for nid in node_map.keys()
                    if nid not in connected_ids and nid not in selected_ids and int(node_map[nid].get("_priority", 0)) >= 1
                ]
                isolated_others = [
                    nid for nid in node_map.keys()
                    if nid not in connected_ids and nid not in selected_ids and nid not in isolated_preferred
                ]

                label_caps = {
                    "CodeFile": 18,
                    "WikiStory": 12,
                    "BusinessTerm": 8,
                    "InBusinessEntity": 10,
                }
                label_counts: Dict[str, int] = {}

                def _can_take(nid: str) -> bool:
                    label = str(node_map[nid].get("label", ""))
                    cap = label_caps.get(label, 6)
                    return label_counts.get(label, 0) < cap

                def _take_node(nid: str) -> bool:
                    if nid in selected_ids:
                        return False
                    if not _can_take(nid):
                        return False
                    selected_ids.append(nid)
                    label = str(node_map[nid].get("label", ""))
                    label_counts[label] = label_counts.get(label, 0) + 1
                    return True

                # 已加入术语本体时计数
                if selected_ids:
                    term_label = str(node_map[selected_ids[0]].get("label", ""))
                    label_counts[term_label] = label_counts.get(term_label, 0) + 1

                connected_others.sort(
                    key=lambda nid: (
                        str(node_map[nid].get("label", "")) != "InBusinessEntity",
                        _rank_node_id(nid),
                    )
                )

                for group in (connected_preferred, connected_others, isolated_preferred, isolated_others):
                    if group is connected_others:
                        iterable = group
                    else:
                        iterable = sorted(group, key=_rank_node_id)
                    for nid in iterable:
                        _take_node(nid)
                        if len(selected_ids) >= 24:
                            break
                    if len(selected_ids) >= 24:
                        break

                # 兜底：如果图里节点充足，保证快照至少有 8 个节点（避免过度筛选后过于稀疏）
                if len(selected_ids) < 8 and len(node_map) >= 8:
                    remaining_ids = [
                        nid for nid in sorted(node_map.keys(), key=_rank_node_id)
                        if nid not in selected_ids
                    ]
                    for nid in remaining_ids:
                        selected_ids.append(nid)
                        if len(selected_ids) >= 8:
                            break

                keep_ids = set(selected_ids[:24])
                nodes = []
                for nid in selected_ids[:24]:
                    node_view = dict(node_map[nid])
                    node_view.pop("_priority", None)
                    nodes.append(node_view)

                edges = []
                for edge in filtered_edges:
                    src = str(edge.get("source", ""))
                    dst = str(edge.get("target", ""))
                    if src in keep_ids and dst in keep_ids:
                        edges.append(edge)

                term_snapshots.append({
                    "term_id": center_id,
                    "term_name": center.get("name", center_id),
                    "term_label": center.get("label", ""),
                    "support": float(center.get("term_support", center.get("score", 0.0))),
                    "hops": snapshot_hops,
                    "nodes": nodes,
                    "edges": edges[:80],
                    "node_count": len(nodes),
                    "edge_count": len(edges),
                })

            # 4) 代码 RepoMap 快照（轻量）
            def _to_record_path_default(node_id: str, node_data: Dict[str, Any]) -> str:
                path = str(node_data.get("path", "")).replace("\\", "/")
                if path:
                    for prefix in ("C:/Git/record/", "C:\\Git\\record\\", "c:/git/record/", "c:\\git\\record\\"):
                        if path.lower().startswith(prefix.lower()):
                            path = path[len(prefix):]
                            break
                    return f"record/{path}".replace("//", "/")
                if node_id.startswith("file:"):
                    return f"record/{node_id[5:]}".replace("//", "/")
                return ""

            code_repomap_snapshots = []
            code_snapshot_candidates = [r for r in reranked if r.get("label") == "CodeFile"]
            if not code_snapshot_candidates:
                code_snapshot_candidates = [r for r in ordered_results if r.get("label") == "CodeFile"]
            top_code_nodes = code_snapshot_candidates[:3]
            for code in top_code_nodes:
                code_id = str(code.get("id", ""))
                if not code_id:
                    continue
                code_data = qm.gm.get_node(code_id) or {}
                code_graph = qm.gm.get_neighbors(code_id, max_hops=1, max_size=80)
                outline = []
                for n in (code_graph or {}).get("nodes", []):
                    n_label = str(n.get("label", ""))
                    if n_label in {"LogicNode", "ArchNode", "ConfigField", "SpecRule", "SpecEvidence"}:
                        outline.append({
                            "label": n_label,
                            "name": n.get("name", n.get("id", "")),
                        })
                outline = outline[:10]
                code_repomap_snapshots.append({
                    "code_id": code_id,
                    "code_name": code.get("name", code_id),
                    "record_path": _to_record_path_default(code_id, code_data) or code_id,
                    "score": float(code.get("score", 0.0)),
                    "outline": outline,
                    "outline_count": len(outline),
                })

            readable_graph_cards = []
            for s in term_snapshots[:3]:
                term_name = str(s.get("term_name", ""))
                nodes = s.get("nodes") or []
                edges = s.get("edges") or []
                id_to_node = {str(n.get("id", "")): n for n in nodes}

                def _edge_rank(edge: Dict[str, Any]) -> tuple:
                    src = id_to_node.get(str(edge.get("source", "")), {})
                    dst = id_to_node.get(str(edge.get("target", "")), {})
                    src_label = str(src.get("label", ""))
                    dst_label = str(dst.get("label", ""))
                    src_term = str(src.get("name", "")) == term_name
                    dst_term = str(dst.get("name", "")) == term_name
                    touches_term = src_term or dst_term
                    label_pair = {src_label, dst_label}
                    if touches_term and "WikiStory" in label_pair:
                        pri = 0
                    elif touches_term and "CodeFile" in label_pair:
                        pri = 1
                    elif touches_term:
                        pri = 2
                    elif "WikiStory" in label_pair and "CodeFile" in label_pair:
                        pri = 3
                    else:
                        pri = 4
                    return (pri, str(edge.get("type", "")), str(edge.get("source", "")), str(edge.get("target", "")))

                top_edges = sorted(edges, key=_edge_rank)[:10]
                edge_statements = []
                seen_stmt = set()
                for e in top_edges:
                    src = id_to_node.get(str(e.get("source", "")), {})
                    dst = id_to_node.get(str(e.get("target", "")), {})
                    if not src or not dst:
                        continue
                    stmt = {
                        "source": {
                            "id": str(src.get("id", "")),
                            "label": str(src.get("label", "")),
                            "name": str(src.get("name", src.get("id", ""))),
                        },
                        "relation": str(e.get("type", "related")),
                        "target": {
                            "id": str(dst.get("id", "")),
                            "label": str(dst.get("label", "")),
                            "name": str(dst.get("name", dst.get("id", ""))),
                        },
                    }
                    sig = (
                        stmt["source"]["id"],
                        stmt["relation"],
                        stmt["target"]["id"],
                    )
                    if sig in seen_stmt:
                        continue
                    seen_stmt.add(sig)
                    edge_statements.append(stmt)

                readable_graph_cards.append({
                    "center": {
                        "name": term_name,
                        "id": str(s.get("term_id", "")),
                        "support": float(s.get("support", 0.0)),
                    },
                    "counts": {
                        "nodes": int(s.get("node_count", 0)),
                        "edges": int(s.get("edge_count", 0)),
                    },
                    "evidence_triplets": edge_statements,
                })

            readable = {
                "summary": {
                    "query": req.query,
                    "effective_mode": effective_mode,
                    "total_results": len(results),
                    "core_result_count": len(core_results),
                    "tail_index_count": len(tail_index),
                    "business_snapshot_count": len(term_snapshots),
                    "code_snapshot_count": len(code_repomap_snapshots),
                },
                "highlights": [
                    {
                        "rank": int(r.get("rank", 0)),
                        "label": str(r.get("label", "")),
                        "name": r.get("name", ""),
                        "score": float(r.get("score", 0.0)),
                    }
                    for r in core_results[:8]
                ],
                "graph_cards": readable_graph_cards,
                "index_refs": [
                    {
                        "rank": int(r.get("rank", 0)),
                        "label": str(r.get("label", "")),
                        "name": r.get("name", ""),
                        "score": float(r.get("score", 0.0)),
                    }
                    for r in tail_index[:8]
                ],
                "business_snapshots": [
                    {
                        "term": s.get("term_name", ""),
                        "support": float(s.get("support", 0.0)),
                        "hops": int(s.get("hops", 0)),
                        "node_count": int(s.get("node_count", 0)),
                        "edge_count": int(s.get("edge_count", 0)),
                        "focus_nodes": [
                            {
                                "label": n.get("label", ""),
                                "name": n.get("name", ""),
                            }
                            for n in sorted(
                                (s.get("nodes") or []),
                                key=lambda n: (
                                    {
                                        "BusinessTerm": 0,
                                        "WikiStory": 1,
                                        "CodeFile": 2,
                                        "InBusinessEntity": 3,
                                    }.get(str(n.get("label", "")), 9),
                                    str(n.get("name", "")),
                                ),
                            )[:8]
                        ],
                    }
                    for s in term_snapshots[:3]
                ],
                "code_snapshots": [
                    {
                        "code": c.get("code_name", ""),
                        "path": c.get("record_path", ""),
                        "score": float(c.get("score", 0.0)),
                        "outline_count": int(c.get("outline_count", 0)),
                        "outline_preview": [
                            {
                                "label": o.get("label", ""),
                                "name": o.get("name", ""),
                            }
                            for o in (c.get("outline") or [])[:6]
                        ],
                    }
                    for c in code_repomap_snapshots[:3]
                ],
            }

            elapsed_ms = int((time.time() - t0) * 1000)
            quick_expansions = []
            for s in term_snapshots[:3]:
                nodes = s.get("nodes") or []
                edges = s.get("edges") or []
                quick_expansions.append({
                    "center": {
                        "id": str(s.get("term_id", "")),
                        "name": s.get("term_name", ""),
                        "label": s.get("term_label", ""),
                        "support": float(s.get("support", 0.0)),
                    },
                    "hops": int(s.get("hops", 0)),
                    "node_count": int(s.get("node_count", 0)),
                    "edge_count": int(s.get("edge_count", 0)),
                    "nodes_preview": [
                        {
                            "id": str(n.get("id", "")),
                            "label": str(n.get("label", "")),
                            "name": n.get("name", ""),
                        }
                        for n in nodes[:8]
                    ],
                    "edges_preview": [
                        {
                            "source": str(e.get("source", "")),
                            "target": str(e.get("target", "")),
                            "relationship": str(e.get("type", e.get("relationship", "related"))),
                        }
                        for e in edges[:10]
                    ],
                })

            response_payload = {
                "results": results,
                "core_results": core_results,
                "quick_expansions": quick_expansions,
                "index_refs": tail_index,
                "elapsed_ms": elapsed_ms,
                "mode": requested_mode,
                "requested_mode": requested_mode,
                "effective_mode": effective_mode,
                "query": req.query,
                "result_type": "list",
                "reorder_meta": {
                    "overfetch_k": overfetch_k,
                    "focus_k": focus_k,
                    "full_hits": len(full_hits),
                    "code_hits": len(code_hits),
                    "term_hits": len(term_hits),
                    "final_candidates": len(reranked),
                    "ignored_label_filter": ignored_label_filter,
                    "excluded_core_labels": sorted(excluded_core_labels),
                    "core_result_count": len(core_results),
                    "tail_index_count": len(tail_index),
                    "core_score_threshold": core_score_threshold,
                    "include_raw_graph": bool(req.include_raw_graph),
                },
                "readable": readable,
            }
            if req.include_raw_graph:
                response_payload["raw_graph"] = {
                    "term_snapshots": term_snapshots,
                    "code_repomap_snapshots": code_repomap_snapshots,
                }
                response_payload["term_snapshots"] = term_snapshots
                response_payload["code_repomap_snapshots"] = code_repomap_snapshots
            return response_payload
        elif req.depth >= 1:
            result = retriever.deep_retrieve(
                query=req.query,
                top_k=req.top_k,
                mode="hybrid",
                label_filter=req.label_filter
            )
            if req.depth == 1:
                result["ppr_expanded"] = []
                result["paths"] = []
            elapsed_ms = int((time.time() - t0) * 1000)
            return {
                "results": result,
                "elapsed_ms": elapsed_ms,
                "mode": requested_mode,
                "requested_mode": requested_mode,
                "effective_mode": effective_mode,
                "query": req.query,
                "result_type": "narrative"
            }
        else:
            results = retriever.retrieve(
                query=req.query,
                top_k=req.top_k,
                mode="hybrid",
                label_filter=req.label_filter
            )

        elapsed_ms = int((time.time() - t0) * 1000)
        return {
            "results": results,
            "elapsed_ms": elapsed_ms,
            "mode": requested_mode,
            "requested_mode": requested_mode,
            "effective_mode": effective_mode,
            "query": req.query,
            "result_type": "list"
        }
    except Exception as e:
        logger.error(f"Query lab search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query-lab/sql")
async def query_lab_sql(req: QueryLabSqlRequest):
    """查询实验 - 只读 SQL 查询"""
    import time
    import sqlite3
    import re
    from pathlib import Path
    from src.config import get_settings

    # 安全检查
    sql_upper = req.sql.strip().upper()
    if not any(sql_upper.startswith(kw) for kw in ["SELECT", "PRAGMA", "EXPLAIN"]):
        raise HTTPException(status_code=400, detail="只允许 SELECT / PRAGMA / EXPLAIN 语句")

    # 确定数据库路径
    settings = get_settings()
    data_dir = Path(settings.paths.data_dir)
    if req.db == "snippets":
        db_path = data_dir / "code_snippets.db"
    else:
        db_path = data_dir / "kg_graph.db"

    if not db_path.exists():
        raise HTTPException(status_code=404, detail=f"数据库不存在: {req.db}")

    # 自动添加 LIMIT（如果没有）
    sql = req.sql.strip()
    if not re.search(r'\bLIMIT\b', sql, re.IGNORECASE):
        sql += " LIMIT 500"

    t0 = time.time()
    try:
        # 只读连接
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5.0)
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(sql)

        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()

        elapsed_ms = int((time.time() - t0) * 1000)
        return {
            "columns": columns,
            "rows": rows,
            "row_count": len(rows),
            "elapsed_ms": elapsed_ms,
            "db": req.db
        }
    except sqlite3.Error as e:
        raise HTTPException(status_code=400, detail=f"SQL 错误: {str(e)}")
    except Exception as e:
        logger.error(f"SQL query error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query-lab/neighbors")
async def query_lab_neighbors(req: QueryLabNeighborsRequest):
    """查询实验 - 邻域查询"""
    import time
    qm = get_question_manager()

    t0 = time.time()
    try:
        hops = max(1, min(3, req.hops))
        result = qm.gm.get_neighbors(req.node_id, max_hops=hops, max_size=100)

        for n in result.get("nodes", []):
            n.pop("_embedding", None)

        elapsed_ms = int((time.time() - t0) * 1000)
        return {
            "node_id": req.node_id,
            "hops": hops,
            "nodes": [_strip_internal(n) for n in result.get("nodes", [])],
            "edges": [_strip_internal(e) for e in result.get("edges", [])],
            "elapsed_ms": elapsed_ms
        }
    except Exception as e:
        logger.error(f"Neighbors query error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query-lab/path")
async def query_lab_path(req: QueryLabPathRequest):
    """查询实验 - 路径查找"""
    import time
    qm = get_question_manager()

    t0 = time.time()
    try:
        max_depth = max(1, min(8, req.max_depth))
        result = qm.gm.shortest_path_detail(req.source_id, req.target_id, max_depth)

        elapsed_ms = int((time.time() - t0) * 1000)
        result["elapsed_ms"] = elapsed_ms
        return result
    except Exception as e:
        logger.error(f"Path query error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------------------------------------------------------
# Graph Edit HTTP API (aligned with stdio MCP tools)
# -------------------------------------------------------------------------

class GraphEditRequest(BaseModel):
    """Common request body for graph edit operations."""
    action: str
    node_id: Optional[str] = None
    label: Optional[str] = None
    properties: Optional[Dict[str, Any]] = None
    source: Optional[str] = None
    target: Optional[str] = None
    relationship: Optional[str] = None
    direction: Optional[str] = None
    dry_run: bool = False
    confirm_token: Optional[str] = None
    # list_by_label / list
    limit: Optional[int] = None
    offset: Optional[int] = None
    filter_key: Optional[str] = None
    filter_value: Optional[str] = None


class MergeNodesRequest(BaseModel):
    source_ids: List[str]
    target_id: str
    delete_sources: bool = False
    dry_run: bool = False
    confirm_token: Optional[str] = None


class TreeOpRequest(BaseModel):
    action: str
    relationship: str
    node_id: Optional[str] = None
    parent_id: Optional[str] = None
    child_id: Optional[str] = None
    new_node_id: Optional[str] = None
    new_node_label: Optional[str] = None
    new_node_properties: Optional[Dict[str, Any]] = None
    dry_run: bool = False
    confirm_token: Optional[str] = None


class QuestionsEditRequest(BaseModel):
    action: str
    question_id: Optional[str] = None
    answer: Optional[str] = None
    severity: Optional[str] = None
    category: Optional[str] = None
    limit: int = 20
    dry_run: bool = False


class VersionRequest(BaseModel):
    action: str
    message: Optional[str] = None
    snapshot_id: Optional[str] = None
    keyword: Optional[str] = None
    limit: int = 30
    dry_run: bool = False
    confirm_token: Optional[str] = None


def _edit_response(result: Dict[str, Any]):
    """Convert a GraphEditService result dict to an HTTP response or raise."""
    if result.get("ok"):
        return result
    err = result.get("error", {})
    status_code = err.get("status", 400)
    raise HTTPException(status_code=status_code, detail={
        "code": err.get("code", "UNKNOWN_ERROR"),
        "message": err.get("message", ""),
        "details": err.get("details", {}),
    })


@router.post("/graph/node/edit")
async def graph_node_edit(req: GraphEditRequest):
    """
    Node CRUD operations: get, create, update, delete, list_by_label.
    Aligned with stdio MCP tool kg_node_edit.
    """
    svc = _get_edit_service()
    result = svc.execute_node_edit(req.model_dump(exclude_none=True))
    return _edit_response(result)


@router.post("/graph/edge/edit")
async def graph_edge_edit(req: GraphEditRequest):
    """
    Edge CRUD operations: get, create, delete, list.
    Aligned with stdio MCP tool kg_edge_edit.
    """
    svc = _get_edit_service()
    result = svc.execute_edge_edit(req.model_dump(exclude_none=True))
    return _edit_response(result)


@router.post("/graph/merge-nodes")
async def graph_merge_nodes(req: MergeNodesRequest):
    """
    Merge multiple source nodes into one target node.
    Aligned with stdio MCP tool kg_merge_nodes.
    """
    svc = _get_edit_service()
    result = svc.execute_merge_nodes(req.model_dump())
    return _edit_response(result)


@router.post("/graph/tree-op")
async def graph_tree_op(req: TreeOpRequest):
    """
    Tree structure operations: remove_and_reparent, insert_between.
    Aligned with stdio MCP tool kg_tree_op.
    """
    svc = _get_edit_service()
    result = svc.execute_tree_op(req.model_dump(exclude_none=True))
    return _edit_response(result)


@router.post("/graph/questions/edit")
async def graph_questions_edit(req: QuestionsEditRequest):
    """
    Question management: stats, list, resolve, dismiss.
    Aligned with stdio MCP tool kg_questions.
    """
    svc = _get_edit_service()
    result = svc.execute_questions(req.model_dump(exclude_none=True))
    return _edit_response(result)


@router.post("/graph/version")
async def graph_version(req: VersionRequest):
    """
    Git-based snapshot management: list, search, save, rollback, delete.
    Aligned with stdio MCP tool kg_version (now git-backed).
    """
    svc = _get_edit_service()
    result = svc.execute_version(req.model_dump(exclude_none=True))
    return _edit_response(result)
