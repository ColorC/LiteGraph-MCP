# -*- coding: utf-8 -*-
"""
统一图编辑服务（stdio MCP + HTTP API 复用）
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Callable

from src.graph.schema import NodeLabel, RelType
from src.tools.question_manager import VALID_CATEGORIES, VALID_STATUSES

logger = logging.getLogger(__name__)


_ALLOWED_LABELS = {x.value for x in NodeLabel}
_ALLOWED_RELATIONSHIPS = {x.value for x in RelType}
_ID_RE = re.compile(r"^[A-Za-z0-9_:\-./\u4e00-\u9fff]{1,256}$")


@dataclass
class GuardrailError(Exception):
    code: str
    message: str
    status: int = 400
    details: Optional[Dict[str, Any]] = None


class GraphEditService:
    def __init__(
        self,
        graph_manager,
        data_dir: Path,
        reload_graph_state: Optional[Callable[[], None]] = None,
    ):
        self.gm = graph_manager
        self.data_dir = Path(data_dir)
        self.db_path = self.data_dir / "kg_graph.db"
        self.reload_graph_state = reload_graph_state

    # -----------------------------
    # Public execution entrypoints
    # -----------------------------

    def execute_node_edit(self, args: Dict[str, Any]) -> Dict[str, Any]:
        action = str(args.get("action", "")).strip()
        dry_run = bool(args.get("dry_run", False))

        try:
            if action == "get":
                node_id = self._validate_node_id(args.get("node_id"), field="node_id")
                data = self.gm.get_node(node_id)
                if not data:
                    raise GuardrailError("NOT_FOUND", f"节点不存在: {node_id}", status=404)
                return self._ok(action, dry_run, {"node": self._strip_internal(data)})

            if action == "create":
                node_id = self._validate_node_id(args.get("node_id"), field="node_id")
                label = self._validate_label(args.get("label"))
                props = self._as_dict(args.get("properties"), "properties")
                self._validate_node_quality(label, props, is_create=True)
                exists = self.gm.get_node(node_id) is not None
                preview = {
                    "node_id": node_id,
                    "label": label,
                    "properties": props,
                    "will_overwrite": exists,
                }
                if dry_run:
                    return self._ok(action, True, {"preview": preview})

                self.gm.add_node(node_id, label, props)
                return self._ok(action, False, {"node_id": node_id, "created": True, "overwrote": exists})

            if action == "update":
                node_id = self._validate_node_id(args.get("node_id"), field="node_id")
                props = self._as_dict(args.get("properties"), "properties")
                if not props:
                    raise GuardrailError("INVALID_ARGUMENT", "update 需要非空 properties")
                existing = self.gm.get_node(node_id)
                if not existing:
                    raise GuardrailError("NOT_FOUND", f"节点不存在: {node_id}", status=404)

                label = str(existing.get("label", ""))
                self._validate_node_quality(label, props, is_create=False)

                if dry_run:
                    return self._ok(action, True, {
                        "preview": {
                            "node_id": node_id,
                            "label": label,
                            "fields_to_change": list(props.keys()),
                            "current_properties": self._strip_internal(existing),
                            "new_properties": props,
                        }
                    })

                self.gm.update_node_properties(node_id, props)
                return self._ok(action, False, {"node_id": node_id, "updated": True, "fields": list(props.keys())})

            if action == "delete":
                node_id = self._validate_node_id(args.get("node_id"), field="node_id")
                self._ensure_high_risk_guard(args, "delete")

                existing = self.gm.get_node(node_id)
                if not existing:
                    raise GuardrailError("NOT_FOUND", f"节点不存在: {node_id}", status=404)

                edges = self.gm.find_edges(node_id, "BOTH")
                if dry_run:
                    return self._ok(action, True, {
                        "preview": {
                            "node_id": node_id,
                            "node_label": existing.get("label", ""),
                            "node_name": existing.get("name", ""),
                            "connected_edges": len(edges),
                            "edge_summary": [
                                f"{e.get('source', '')[:40]}-[{e.get('relationship', '?')}]->{e.get('target', '')[:40]}"
                                for e in edges[:10]
                            ],
                        }
                    })

                self.gm.delete_node(node_id)
                return self._ok(action, False, {"node_id": node_id, "deleted": True, "removed_edges": len(edges)})

            if action == "list_by_label":
                label = self._validate_label(args.get("label"))
                limit = self._clamp_int(args.get("limit", 50), 50, 1, 200)
                offset = self._clamp_int(args.get("offset", 0), 0, 0, 10_000)
                filter_key = str(args.get("filter_key") or "").strip()
                filter_value = str(args.get("filter_value") or "").strip()

                nodes = self.gm.find_nodes_by_label(label)
                if filter_key and filter_value:
                    nodes = [n for n in nodes if filter_value.lower() in str(n.get(filter_key, "")).lower()]

                total = len(nodes)
                page = [self._strip_internal(x) for x in nodes[offset:offset + limit]]
                return self._ok(action, dry_run, {
                    "label": label,
                    "total": total,
                    "offset": offset,
                    "limit": limit,
                    "items": page,
                })

            raise GuardrailError("UNKNOWN_ACTION", f"未知 node_edit action: {action}")

        except GuardrailError as e:
            return self._err(e)
        except Exception as e:
            logger.exception("execute_node_edit failed")
            return self._err(GuardrailError("INTERNAL_ERROR", str(e), status=500))

    def execute_edge_edit(self, args: Dict[str, Any]) -> Dict[str, Any]:
        action = str(args.get("action", "")).strip()
        dry_run = bool(args.get("dry_run", False))

        try:
            if action == "get":
                source = self._validate_node_id(args.get("source"), field="source")
                target = self._validate_node_id(args.get("target"), field="target")
                relationship = args.get("relationship")
                if relationship is not None:
                    relationship = self._validate_relationship(relationship)
                edge = self.gm.get_edge(source, target, relationship)
                if not edge:
                    raise GuardrailError("NOT_FOUND", f"边不存在: {source} -> {target}", status=404)
                return self._ok(action, dry_run, {"edge": self._strip_internal(edge)})

            if action == "create":
                source = self._validate_node_id(args.get("source"), field="source")
                target = self._validate_node_id(args.get("target"), field="target")
                relationship = self._validate_relationship(args.get("relationship"))
                props = self._as_dict(args.get("properties"), "properties")

                self._validate_edge_endpoints(source, target, relationship)

                exists = self.gm.get_edge(source, target, relationship) is not None
                src_node = self.gm.get_node(source)
                tgt_node = self.gm.get_node(target)
                preview = {
                    "source": source,
                    "target": target,
                    "relationship": relationship,
                    "properties": props,
                    "source_exists": src_node is not None,
                    "target_exists": tgt_node is not None,
                    "source_label": (src_node or {}).get("label"),
                    "target_label": (tgt_node or {}).get("label"),
                    "will_overwrite": exists,
                }
                if dry_run:
                    return self._ok(action, True, {"preview": preview})

                self.gm.add_edge(source, target, relationship, props)
                return self._ok(action, False, {
                    "source": source,
                    "target": target,
                    "relationship": relationship,
                    "created": True,
                    "overwrote": exists,
                })

            if action == "delete":
                source = self._validate_node_id(args.get("source"), field="source")
                target = self._validate_node_id(args.get("target"), field="target")
                relationship = args.get("relationship")
                if relationship is not None:
                    relationship = self._validate_relationship(relationship)
                self._ensure_high_risk_guard(args, "delete_edge")

                existing = self.gm.get_edge(source, target, relationship)
                if not existing:
                    raise GuardrailError("NOT_FOUND", f"边不存在: {source} -> {target}", status=404)

                if dry_run:
                    return self._ok(action, True, {
                        "preview": {
                            "source": source,
                            "target": target,
                            "relationship": existing.get("relationship"),
                            "edge_properties": self._strip_internal(existing),
                        }
                    })

                self.gm.delete_edge(source, target, relationship)
                return self._ok(action, False, {
                    "source": source,
                    "target": target,
                    "relationship": existing.get("relationship"),
                    "deleted": True,
                })

            if action == "list":
                node_id = self._validate_node_id(args.get("node_id"), field="node_id")
                direction = str(args.get("direction", "BOTH")).upper().strip()
                if direction not in {"OUT", "IN", "BOTH"}:
                    raise GuardrailError("INVALID_DIRECTION", "direction 必须是 OUT/IN/BOTH")
                relationship = args.get("relationship")
                if relationship is not None:
                    relationship = self._validate_relationship(relationship)

                edges = [self._strip_internal(x) for x in self.gm.find_edges(node_id, direction, relationship)]
                return self._ok(action, dry_run, {
                    "node_id": node_id,
                    "direction": direction,
                    "relationship": relationship,
                    "total": len(edges),
                    "items": edges,
                })

            raise GuardrailError("UNKNOWN_ACTION", f"未知 edge_edit action: {action}")

        except GuardrailError as e:
            return self._err(e)
        except Exception as e:
            logger.exception("execute_edge_edit failed")
            return self._err(GuardrailError("INTERNAL_ERROR", str(e), status=500))

    def execute_merge_nodes(self, args: Dict[str, Any]) -> Dict[str, Any]:
        dry_run = bool(args.get("dry_run", False))
        try:
            source_ids = args.get("source_ids", [])
            if not isinstance(source_ids, list) or not source_ids:
                raise GuardrailError("INVALID_ARGUMENT", "需要 source_ids 数组")
            source_ids = [self._validate_node_id(x, field="source_ids") for x in source_ids]

            target_id = self._validate_node_id(args.get("target_id"), field="target_id")
            delete_sources = bool(args.get("delete_sources", False))
            self._ensure_high_risk_guard(args, "merge_nodes")

            target_data = self.gm.get_node(target_id)
            if not target_data:
                raise GuardrailError("NOT_FOUND", f"目标节点不存在: {target_id}", status=404)

            impacts = []
            total_edges = 0
            for src_id in source_ids:
                src_data = self.gm.get_node(src_id)
                if not src_data or src_id == target_id:
                    impacts.append({"source_id": src_id, "status": "skip", "reason": "不存在或等于目标"})
                    continue
                out_edges = self.gm.find_edges(src_id, "OUT")
                in_edges = self.gm.find_edges(src_id, "IN")
                edge_count = len(out_edges) + len(in_edges)
                total_edges += edge_count
                impacts.append({
                    "source_id": src_id,
                    "source_label": src_data.get("label", "?"),
                    "source_name": src_data.get("name", ""),
                    "out_edges": len(out_edges),
                    "in_edges": len(in_edges),
                })

            if dry_run:
                return self._ok("merge_nodes", True, {
                    "preview": {
                        "target_id": target_id,
                        "target_label": target_data.get("label", ""),
                        "target_name": target_data.get("name", ""),
                        "delete_sources": delete_sources,
                        "source_impacts": impacts,
                        "total_edges_to_redirect": total_edges,
                    }
                })

            merge_result = self.gm.merge_nodes(source_ids, target_id, delete_sources)
            return self._ok("merge_nodes", False, {
                "target_id": target_id,
                "source_ids": source_ids,
                "delete_sources": delete_sources,
                "result": merge_result,
            })

        except GuardrailError as e:
            return self._err(e)
        except Exception as e:
            logger.exception("execute_merge_nodes failed")
            return self._err(GuardrailError("INTERNAL_ERROR", str(e), status=500))

    def execute_tree_op(self, args: Dict[str, Any]) -> Dict[str, Any]:
        action = str(args.get("action", "")).strip()
        rel = self._validate_relationship(args.get("relationship"))
        dry_run = bool(args.get("dry_run", False))

        try:
            self._ensure_high_risk_guard(args, "tree_op")

            if action == "remove_and_reparent":
                node_id = self._validate_node_id(args.get("node_id"), field="node_id")
                if not self.gm.has_node(node_id):
                    raise GuardrailError("NOT_FOUND", f"节点不存在: {node_id}", status=404)

                parents = [e["source"] for e in self.gm.find_edges(node_id, direction="IN", relationship=rel)]
                children = [e["target"] for e in self.gm.find_edges(node_id, direction="OUT", relationship=rel)]

                if dry_run:
                    return self._ok(action, True, {
                        "preview": {
                            "node_id": node_id,
                            "relationship": rel,
                            "parents": parents,
                            "children": children,
                            "new_edges_to_create": [f"{p} -[{rel}]-> {c}" for p in parents for c in children],
                        }
                    })

                reparented = 0
                for parent in parents:
                    for child in children:
                        self.gm.add_edge(parent, child, rel)
                        reparented += 1
                self.gm.delete_node(node_id)
                return self._ok(action, False, {
                    "removed_node": node_id,
                    "reparented_count": reparented,
                    "parent_count": len(parents),
                    "child_count": len(children),
                })

            if action == "insert_between":
                parent_id = self._validate_node_id(args.get("parent_id"), field="parent_id")
                child_id = self._validate_node_id(args.get("child_id"), field="child_id")
                new_id = self._validate_node_id(args.get("new_node_id"), field="new_node_id")
                new_label = self._validate_label(args.get("new_node_label"))
                new_props = self._as_dict(args.get("new_node_properties"), "new_node_properties")
                self._validate_node_quality(new_label, new_props, is_create=True)

                if not self.gm.has_edge(parent_id, child_id, rel):
                    raise GuardrailError("NOT_FOUND", f"边不存在: {parent_id} -[{rel}]-> {child_id}", status=404)

                if dry_run:
                    return self._ok(action, True, {
                        "preview": {
                            "parent_id": parent_id,
                            "child_id": child_id,
                            "new_node": {"id": new_id, "label": new_label, "properties": new_props},
                            "relationship": rel,
                            "steps": [
                                f"删除边: {parent_id} -[{rel}]-> {child_id}",
                                f"创建节点: {new_id} ({new_label})",
                                f"创建边: {parent_id} -[{rel}]-> {new_id}",
                                f"创建边: {new_id} -[{rel}]-> {child_id}",
                            ],
                        }
                    })

                self.gm.delete_edge(parent_id, child_id, rel)
                self.gm.add_node(new_id, new_label, new_props)
                self.gm.add_edge(parent_id, new_id, rel)
                self.gm.add_edge(new_id, child_id, rel)
                return self._ok(action, False, {
                    "parent_id": parent_id,
                    "child_id": child_id,
                    "new_node_id": new_id,
                    "relationship": rel,
                    "inserted": True,
                })

            raise GuardrailError("UNKNOWN_ACTION", f"未知 tree_op action: {action}")

        except GuardrailError as e:
            return self._err(e)
        except Exception as e:
            logger.exception("execute_tree_op failed")
            return self._err(GuardrailError("INTERNAL_ERROR", str(e), status=500))

    def execute_questions(self, args: Dict[str, Any]) -> Dict[str, Any]:
        action = str(args.get("action", "")).strip()
        dry_run = bool(args.get("dry_run", False))

        try:
            if action == "stats":
                questions = self.gm.find_nodes_by_label("Question")
                from collections import Counter
                by_status = Counter()
                by_cat = Counter()
                by_sev = Counter()
                for q in questions:
                    by_status[q.get("status", "?")] += 1
                    by_cat[q.get("category", "?")] += 1
                    by_sev[q.get("severity", "?")] += 1
                return self._ok(action, dry_run, {
                    "total": len(questions),
                    "by_status": dict(by_status),
                    "by_category": dict(by_cat),
                    "by_severity": dict(by_sev),
                })

            if action == "list":
                severity = (args.get("severity") or "").strip() or None
                category = (args.get("category") or "").strip() or None
                if category and category not in VALID_CATEGORIES:
                    raise GuardrailError("INVALID_CATEGORY", f"非法 category: {category}")

                limit = self._clamp_int(args.get("limit", 20), 20, 1, 200)
                questions = self.gm.find_nodes_by_label("Question")
                filtered = []
                for q in questions:
                    if q.get("status", "pending") != "pending":
                        continue
                    if severity and q.get("severity") != severity:
                        continue
                    if category and q.get("category") != category:
                        continue
                    filtered.append(self._strip_internal(q))

                return self._ok(action, dry_run, {
                    "total": len(filtered),
                    "severity": severity,
                    "category": category,
                    "items": filtered[:limit],
                    "limit": limit,
                })

            if action == "resolve":
                qid = self._validate_node_id(args.get("question_id"), field="question_id")
                answer = str(args.get("answer") or "").strip()
                if not answer or len(answer) < 2:
                    raise GuardrailError("LOW_QUALITY_PAYLOAD", "resolve 需要非空 answer（至少2字符）")
                if len(answer) > 5000:
                    raise GuardrailError("LOW_QUALITY_PAYLOAD", "answer 过长（>5000）")

                existing = self.gm.get_node(qid)
                if not existing:
                    raise GuardrailError("NOT_FOUND", f"Question 不存在: {qid}", status=404)

                if dry_run:
                    return self._ok(action, True, {
                        "preview": {
                            "question_id": qid,
                            "answer": answer,
                            "current_data": self._strip_internal(existing),
                        }
                    })

                self.gm.update_node_properties(qid, {"status": "resolved", "answer": answer})
                return self._ok(action, False, {"question_id": qid, "resolved": True})

            if action == "dismiss":
                qid = self._validate_node_id(args.get("question_id"), field="question_id")
                existing = self.gm.get_node(qid)
                if not existing:
                    raise GuardrailError("NOT_FOUND", f"Question 不存在: {qid}", status=404)

                if dry_run:
                    return self._ok(action, True, {
                        "preview": {
                            "question_id": qid,
                            "current_data": self._strip_internal(existing),
                        }
                    })

                self.gm.update_node_properties(qid, {"status": "dismissed"})
                return self._ok(action, False, {"question_id": qid, "dismissed": True})

            raise GuardrailError("UNKNOWN_ACTION", f"未知 questions action: {action}")

        except GuardrailError as e:
            return self._err(e)
        except Exception as e:
            logger.exception("execute_questions failed")
            return self._err(GuardrailError("INTERNAL_ERROR", str(e), status=500))

    def execute_version(self, args: Dict[str, Any]) -> Dict[str, Any]:
        action = str(args.get("action", "")).strip()
        dry_run = bool(args.get("dry_run", False))

        try:
            self._ensure_git_repo_for_db()

            if action == "list":
                entries = self._git_log_for_db(limit=self._clamp_int(args.get("limit", 30), 30, 1, 200))
                return self._ok(action, dry_run, {"items": entries, "total": len(entries)})

            if action == "search":
                keyword = str(args.get("keyword") or "").strip()
                if not keyword:
                    raise GuardrailError("INVALID_ARGUMENT", "search 需要 keyword")
                entries = self._git_log_for_db(keyword=keyword, limit=self._clamp_int(args.get("limit", 50), 50, 1, 200))
                return self._ok(action, dry_run, {"keyword": keyword, "items": entries, "total": len(entries)})

            if action == "save":
                message = str(args.get("message") or "").strip()
                if not message:
                    raise GuardrailError("INVALID_ARGUMENT", "save 需要 message")
                if len(message) > 240:
                    raise GuardrailError("LOW_QUALITY_PAYLOAD", "message 过长（>240）")

                if dry_run:
                    return self._ok(action, True, {
                        "preview": {
                            "message": message,
                            "db_path": str(self.db_path),
                            "node_count": self.gm.node_count(),
                            "edge_count": self.gm.edge_count(),
                        }
                    })

                commit_info = self._git_snapshot_save(message)
                return self._ok(action, False, {"snapshot": commit_info})

            if action == "rollback":
                commit_id = str(args.get("snapshot_id") or "").strip()
                if not commit_id:
                    raise GuardrailError("INVALID_ARGUMENT", "rollback 需要 snapshot_id")
                self._ensure_high_risk_guard(args, "rollback")

                if dry_run:
                    return self._ok(action, True, {
                        "preview": {
                            "snapshot_id": commit_id,
                            "note": "回退前会尝试自动创建保护快照（若有变更）",
                            "current_nodes": self.gm.node_count(),
                            "current_edges": self.gm.edge_count(),
                        }
                    })

                self._git_snapshot_save(f"[auto-save before rollback to {commit_id}]", allow_no_changes=True)
                repo_root = self._git_repo_root_for_db()
                rel_db = self._db_relpath(repo_root)
                self._run_git(repo_root, ["checkout", commit_id, "--", rel_db])
                self._reload_graph_state()
                return self._ok(action, False, {
                    "rolled_back_to": commit_id,
                    "node_count": self.gm.node_count(),
                    "edge_count": self.gm.edge_count(),
                })

            if action == "delete":
                snapshot_id = str(args.get("snapshot_id") or "").strip()
                if not snapshot_id:
                    raise GuardrailError("INVALID_ARGUMENT", "delete 需要 snapshot_id")
                self._ensure_high_risk_guard(args, "delete_snapshot")

                if dry_run:
                    return self._ok(action, True, {
                        "preview": {
                            "snapshot_id": snapshot_id,
                            "semantic": "git 历史不物理删除，delete 会记录 archived 标记提交",
                        }
                    })

                archive_message = f"kg-snapshot-archive: {snapshot_id}"
                commit_info = self._git_snapshot_save(archive_message, allow_no_changes=True, touch_archive_note=True)
                return self._ok(action, False, {
                    "snapshot_id": snapshot_id,
                    "archived": True,
                    "snapshot": commit_info,
                })

            raise GuardrailError("UNKNOWN_ACTION", f"未知 version action: {action}")

        except GuardrailError as e:
            return self._err(e)
        except Exception as e:
            logger.exception("execute_version failed")
            return self._err(GuardrailError("INTERNAL_ERROR", str(e), status=500))

    def execute_apply(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        action = str(action or "").strip()
        params = params if isinstance(params, dict) else {}
        dry_run = bool(params.get("dry_run", False))

        try:
            from src.skills.story_linkage.tools import StoryLinkageTools

            if action == "add_edge":
                source = self._validate_node_id(params.get("source"), field="source")
                target = self._validate_node_id(params.get("target"), field="target")
                relationship = self._validate_relationship(params.get("relationship", "IMPLEMENTED_IN"))
                self._validate_edge_endpoints(source, target, relationship)

                if dry_run:
                    return self._ok(action, True, {
                        "preview": {
                            "source": source,
                            "target": target,
                            "relationship": relationship,
                            "note": "将调用 StoryLinkageTools.add_verified_linkage",
                        }
                    })

                tools = StoryLinkageTools()
                msg = tools.add_verified_linkage(story_id=source, code_node_id=target, relationship=relationship)
                return self._ok(action, False, {
                    "source": source,
                    "target": target,
                    "relationship": relationship,
                    "message": msg,
                })

            if action == "enrich_logic":
                source = self._validate_node_id(params.get("source"), field="source")
                description = str(params.get("description") or "").strip()
                if len(description) < 4:
                    raise GuardrailError("LOW_QUALITY_PAYLOAD", "description 过短（至少4字符）")
                if len(description) > 5000:
                    raise GuardrailError("LOW_QUALITY_PAYLOAD", "description 过长（>5000）")

                if dry_run:
                    return self._ok(action, True, {
                        "preview": {
                            "source": source,
                            "description": description,
                            "note": "将调用 StoryLinkageTools.enrich_graph_logic",
                        }
                    })

                tools = StoryLinkageTools()
                msg = tools.enrich_graph_logic(story_id=source, logic_description=description)
                return self._ok(action, False, {
                    "source": source,
                    "message": msg,
                })

            raise GuardrailError("UNKNOWN_ACTION", f"Unknown apply action: {action}")

        except GuardrailError as e:
            return self._err(e)
        except Exception as e:
            logger.exception("execute_apply failed")
            return self._err(GuardrailError("INTERNAL_ERROR", str(e), status=500))

    # -----------------------------
    # Guardrails / validators
    # -----------------------------

    def _validate_node_id(self, value: Any, field: str = "node_id") -> str:
        node_id = str(value or "").strip()
        if not node_id:
            raise GuardrailError("INVALID_NODE_ID", f"{field} 不能为空")
        if len(node_id) > 256:
            raise GuardrailError("INVALID_NODE_ID", f"{field} 过长")
        if not _ID_RE.match(node_id):
            raise GuardrailError("INVALID_NODE_ID", f"{field} 含非法字符: {node_id}")
        return node_id

    def _validate_label(self, value: Any) -> str:
        label = str(value or "").strip()
        if not label:
            raise GuardrailError("INVALID_LABEL", "label 不能为空")
        if label not in _ALLOWED_LABELS:
            raise GuardrailError("INVALID_LABEL", f"不允许的 label: {label}", details={"allowed_labels": sorted(_ALLOWED_LABELS)})
        return label

    def _validate_relationship(self, value: Any) -> str:
        relationship = str(value or "").strip()
        if not relationship:
            raise GuardrailError("INVALID_RELATIONSHIP", "relationship 不能为空")
        if relationship not in _ALLOWED_RELATIONSHIPS:
            raise GuardrailError(
                "INVALID_RELATIONSHIP",
                f"不允许的 relationship: {relationship}",
                details={"allowed_relationships": sorted(_ALLOWED_RELATIONSHIPS)},
            )
        return relationship

    def _validate_node_quality(self, label: str, props: Dict[str, Any], is_create: bool) -> None:
        name = str(props.get("name") or "").strip()
        description = str(props.get("description") or "").strip()
        question = str(props.get("question") or "").strip()

        if is_create:
            if label == "Question":
                if len(question) < 4:
                    raise GuardrailError("LOW_QUALITY_PAYLOAD", "Question 节点需要 question（至少4字符）")
            else:
                if not name and not description:
                    raise GuardrailError("LOW_QUALITY_PAYLOAD", "create 需要 name 或 description 至少其一")

        if name and len(name) > 200:
            raise GuardrailError("LOW_QUALITY_PAYLOAD", "name 过长（>200）")
        if description and len(description) > 5000:
            raise GuardrailError("LOW_QUALITY_PAYLOAD", "description 过长（>5000）")
        if question and len(question) > 5000:
            raise GuardrailError("LOW_QUALITY_PAYLOAD", "question 过长（>5000）")

        if "status" in props:
            status = str(props.get("status") or "").strip()
            if status and status not in VALID_STATUSES and status != "resolved":
                raise GuardrailError("LOW_QUALITY_PAYLOAD", f"非法 status: {status}")

        if "category" in props:
            category = str(props.get("category") or "").strip()
            if category and category not in VALID_CATEGORIES:
                raise GuardrailError("LOW_QUALITY_PAYLOAD", f"非法 category: {category}")

    def _validate_edge_endpoints(self, source: str, target: str, relationship: str) -> None:
        src = self.gm.get_node(source)
        tgt = self.gm.get_node(target)
        if not src or not tgt:
            raise GuardrailError(
                "ENDPOINT_NOT_FOUND",
                "source/target 节点必须存在",
                details={"source_exists": bool(src), "target_exists": bool(tgt)},
            )

        src_label = str(src.get("label", ""))
        tgt_label = str(tgt.get("label", ""))
        if src_label not in _ALLOWED_LABELS or tgt_label not in _ALLOWED_LABELS:
            raise GuardrailError("INVALID_LABEL", "source/target 的 label 非法")

        # 可选端点约束（默认关闭，生产可开启）
        if self._bool_env("open_graph_ENFORCE_RELATIONSHIP_ENDPOINTS", default=False):
            allowed_pairs = self._load_relationship_rules().get(relationship, set())
            if allowed_pairs and (src_label, tgt_label) not in allowed_pairs:
                raise GuardrailError(
                    "INVALID_RELATIONSHIP_ENDPOINT",
                    f"关系 {relationship} 不允许 {src_label} -> {tgt_label}",
                    details={"allowed_pairs": sorted([f"{a}->{b}" for a, b in allowed_pairs])},
                )

    def _ensure_high_risk_guard(self, args: Dict[str, Any], op: str) -> None:
        if bool(args.get("dry_run", False)):
            return

        require_token = self._bool_env("open_graph_REQUIRE_CONFIRM_TOKEN", default=False)
        if not require_token:
            return

        expected = str(os.environ.get("open_graph_CONFIRM_TOKEN") or "").strip()
        given = str(args.get("confirm_token") or "").strip()
        if not expected:
            raise GuardrailError("CONFIRM_TOKEN_MISCONFIG", "服务端未配置 open_graph_CONFIRM_TOKEN", status=500)
        if given != expected:
            raise GuardrailError(
                "CONFIRM_REQUIRED",
                f"高风险操作 {op} 需要有效 confirm_token",
                status=403,
            )

    # -----------------------------
    # Git snapshot backend
    # -----------------------------

    def _ensure_git_repo_for_db(self) -> None:
        if not self.db_path.exists():
            raise GuardrailError("DB_NOT_FOUND", f"数据库不存在: {self.db_path}", status=404)
        try:
            _ = self._git_repo_root_for_db()
        except GuardrailError:
            raise
        except Exception as e:
            raise GuardrailError("GIT_NOT_AVAILABLE", f"无法定位 git 仓库: {e}", status=400)

    def _git_repo_root_for_db(self) -> Path:
        out = self._run_git(self.data_dir, ["rev-parse", "--show-toplevel"])
        root = Path(out.strip())
        if not root.exists():
            raise GuardrailError("GIT_NOT_AVAILABLE", f"git 根目录不存在: {root}", status=400)
        return root

    def _db_relpath(self, repo_root: Path) -> str:
        try:
            return str(self.db_path.resolve().relative_to(repo_root.resolve())).replace("\\", "/")
        except Exception:
            raise GuardrailError("GIT_PATH_ERROR", "kg_graph.db 不在 git 仓库内，无法做 git 快照", status=400)

    def _git_log_for_db(self, keyword: str = "", limit: int = 30) -> list[Dict[str, Any]]:
        repo_root = self._git_repo_root_for_db()
        rel_db = self._db_relpath(repo_root)
        pretty = "%H\t%cI\t%s"
        cmd = ["log", f"--pretty=format:{pretty}", f"-n{limit}"]
        if keyword:
            cmd.extend(["--grep", keyword, "-i"])
        cmd.extend(["--", rel_db])

        out = self._run_git(repo_root, cmd, allow_empty=True)
        entries = []
        for line in (out or "").splitlines():
            parts = line.split("\t", 2)
            if len(parts) < 3:
                continue
            commit, commit_time, message = parts
            entries.append({"snapshot_id": commit, "timestamp": commit_time, "message": message})
        return entries

    def _git_snapshot_save(
        self,
        message: str,
        allow_no_changes: bool = False,
        touch_archive_note: bool = False,
    ) -> Dict[str, Any]:
        repo_root = self._git_repo_root_for_db()
        rel_db = self._db_relpath(repo_root)

        archive_note = self.data_dir / ".kg_snapshot_archive.log"
        if touch_archive_note:
            archive_note.parent.mkdir(parents=True, exist_ok=True)
            with open(archive_note, "a", encoding="utf-8") as f:
                ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                f.write(f"{ts}\t{message}\n")

        self._run_git(repo_root, ["add", "--", rel_db])
        if touch_archive_note:
            rel_note = str(archive_note.resolve().relative_to(repo_root.resolve())).replace("\\", "/")
            self._run_git(repo_root, ["add", "--", rel_note])

        porcelain = self._run_git(repo_root, ["status", "--porcelain", "--", rel_db], allow_empty=True)
        has_db_change = bool((porcelain or "").strip())
        has_note_change = False
        if touch_archive_note:
            note_status = self._run_git(repo_root, ["status", "--porcelain", "--", rel_note], allow_empty=True)
            has_note_change = bool((note_status or "").strip())

        if not has_db_change and not has_note_change and not allow_no_changes:
            return {
                "snapshot_id": "",
                "message": "没有检测到 db 变更，未创建新快照",
                "created": False,
            }

        if not has_db_change and not has_note_change and allow_no_changes:
            latest = self._run_git(repo_root, ["rev-parse", "HEAD"], allow_empty=True).strip()
            return {
                "snapshot_id": latest,
                "message": "没有新变更，沿用当前 HEAD",
                "created": False,
            }

        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        commit_msg = f"kg-snapshot: {message} | {ts}"
        self._run_git(repo_root, ["commit", "-m", commit_msg])
        commit_id = self._run_git(repo_root, ["rev-parse", "HEAD"]).strip()
        return {
            "snapshot_id": commit_id,
            "message": commit_msg,
            "created": True,
            "node_count": self.gm.node_count(),
            "edge_count": self.gm.edge_count(),
        }

    def _run_git(self, cwd: Path, args: list[str], allow_empty: bool = False) -> str:
        cmd = ["git", *args]
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip()
            raise GuardrailError("GIT_COMMAND_FAILED", f"git {' '.join(args)} 失败: {err}", status=400)
        out = proc.stdout or ""
        if not out.strip() and not allow_empty:
            return ""
        return out

    def _reload_graph_state(self) -> None:
        if self.reload_graph_state:
            self.reload_graph_state()

    # -----------------------------
    # Utilities
    # -----------------------------

    def _ok(self, action: str, dry_run: bool, data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "ok": True,
            "action": action,
            "dry_run": dry_run,
            "data": data,
        }

    def _err(self, e: GuardrailError) -> Dict[str, Any]:
        return {
            "ok": False,
            "action": "",
            "dry_run": False,
            "error": {
                "code": e.code,
                "message": e.message,
                "details": e.details or {},
                "status": e.status,
            },
        }

    def _as_dict(self, value: Any, field: str) -> Dict[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise GuardrailError("INVALID_ARGUMENT", f"{field} 必须是 object")
        return value

    @staticmethod
    def _strip_internal(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: v for k, v in obj.items() if k != "_embedding"}
        if isinstance(obj, list):
            return [GraphEditService._strip_internal(x) for x in obj]
        return obj

    @staticmethod
    def _clamp_int(value: Any, default: int, min_value: int, max_value: int) -> int:
        try:
            iv = int(value)
        except Exception:
            iv = default
        return max(min_value, min(max_value, iv))

    @staticmethod
    def _bool_env(name: str, default: bool = False) -> bool:
        raw = str(os.environ.get(name, "")).strip().lower()
        if not raw:
            return default
        return raw in {"1", "true", "yes", "on"}

    @staticmethod
    def _load_relationship_rules() -> Dict[str, set[tuple[str, str]]]:
        raw = str(os.environ.get("open_graph_RELATIONSHIP_RULES_JSON") or "").strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
            out: Dict[str, set[tuple[str, str]]] = {}
            if isinstance(parsed, dict):
                for rel, arr in parsed.items():
                    pairs = set()
                    if isinstance(arr, list):
                        for item in arr:
                            if not isinstance(item, dict):
                                continue
                            src = str(item.get("source_label") or "").strip()
                            tgt = str(item.get("target_label") or "").strip()
                            if src and tgt:
                                pairs.add((src, tgt))
                    if pairs:
                        out[str(rel)] = pairs
            return out
        except Exception:
            logger.exception("Failed to parse open_graph_RELATIONSHIP_RULES_JSON")
            return {}


def format_edit_result_text(result: Dict[str, Any]) -> str:
    if result.get("ok"):
        action = result.get("action", "")
        dry_run = bool(result.get("dry_run", False))
        data = result.get("data", {})
        prefix = "[DRY RUN] " if dry_run else ""
        return f"{prefix}{action}\n" + json.dumps(data, ensure_ascii=False, indent=2, default=str)

    err = result.get("error", {}) or {}
    code = err.get("code", "UNKNOWN_ERROR")
    msg = err.get("message", "")
    details = err.get("details") or {}
    if details:
        return f"Error[{code}]: {msg}\n" + json.dumps(details, ensure_ascii=False, indent=2, default=str)
    return f"Error[{code}]: {msg}"
