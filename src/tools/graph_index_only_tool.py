# -*- coding: utf-8 -*-
"""
Graph RAG Index-Only 检索工具

提供精简的索引检索模式，只返回：
- 核心术语集群中的最高匹配度结果
- 文件相对于 Git 的路径 / Wiki wiki 链接 / Wiki story 链接
- 不返回节点的详细描述内容

适用于：
- 急速定位术语和资产
- 作为 Agent 探索的"火花塞"
- 避免长上下文污染
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.retrieval.hybrid_retriever import get_graph_retriever

logger = logging.getLogger(__name__)


# =============================================================================
# 数据结构
# =============================================================================

@dataclass
class IndexMatch:
    """索引匹配结果"""
    term_name: str           # 术语名
    node_type: str           # 节点类型 (CodeFile, BusinessTerm, InBusinessEntity 等)
    asset_link: str          # 资产链接 (Git 路径或Wiki链接)
    confidence: float        # 标准化分数 (0-1)
    confidence_level: str    # 兼容字段
    match_reason: str        # 匹配来源
    raw_score: float = 0.0   # 检索器原始分值


@dataclass
class IndexOnlyResult:
    """Index-only 检索结果"""
    query: str
    high_confidence_matches: List[IndexMatch] = field(default_factory=list)
    low_confidence_matches: List[IndexMatch] = field(default_factory=list)

    def format(self) -> str:
        """格式化为输出字符串（不再分高低组，仅展示标准化分值）。"""
        all_matches = sorted(
            [*self.high_confidence_matches, *self.low_confidence_matches],
            key=lambda m: (-m.confidence, -m.raw_score, m.term_name),
        )

        if not all_matches:
            return "未找到相关结果"

        lines = ["【匹配结果】"]
        for idx, match in enumerate(all_matches, 1):
            lines.append(
                f"- [rank={idx} | score_norm={match.confidence:.3f} | score_raw={match.raw_score:.6f} | source={match.match_reason}] "
                f"{match.term_name} ({match.node_type}): {match.asset_link}"
            )

        return "\n".join(lines)


# =============================================================================
# IndexOnlyRetriever
# =============================================================================

class IndexOnlyRetriever:
    """基于 Hybrid 索引的 Index-only 检索器。"""

    def __init__(
        self,
        git_root: str = "record/",
        wiki_domain: str = "https://xxx.wiki.cn"
    ):
        self.git_root = git_root.rstrip("/")
        self.wiki_domain = wiki_domain.rstrip("/")
        self.retriever = get_graph_retriever()

    def _extract_asset_link(self, node_entry: dict[str, Any]) -> str:
        node_id = str(node_entry.get("id", ""))
        label = str(node_entry.get("label", ""))
        name = str(node_entry.get("name", node_id))

        node_data = self.retriever.gm.get_node(node_id) or {}

        for key in ("wiki_url", "wiki_link", "story_url", "sheet_url"):
            if node_data.get(key):
                return str(node_data[key])

        if label in ("CodeFile", "Folder"):
            path = str(node_data.get("path", "")).replace("\\", "/")
            if path:
                for prefix in ("C:/Git/record/", "C:\\Git\\record\\", "c:/git/record/", "c:\\git\\record\\"):
                    if path.lower().startswith(prefix.lower()):
                        path = path[len(prefix):]
                        break
                return f"{self.git_root}/{path}".replace("//", "/")

            if node_id.startswith("file:"):
                return f"{self.git_root}/{node_id[5:]}".replace("//", "/")
            if node_id.startswith("dir:"):
                return f"{self.git_root}/{node_id[4:]}".replace("//", "/")

        if label in ("BusinessTerm", "InBusinessEntity"):
            # 不生成占位搜索 URL，避免捏造链接
            return node_id

        if label == "Prefab" and node_data.get("path"):
            prefab_path = str(node_data["path"]).replace("\\", "/")
            return f"{self.git_root}/{prefab_path}".replace("//", "/")

        return node_id

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        high_confidence_only: bool = False,
        node_types: Optional[List[str]] = None
    ) -> IndexOnlyResult:
        """使用 Hybrid 检索器（向量+BM25 索引）返回 index-only 结果。"""
        result = IndexOnlyResult(query=query)

        label_filter = node_types[0] if node_types and len(node_types) == 1 else ""
        candidates = self.retriever.retrieve(
            query=query,
            top_k=max(top_k * 5, top_k * 2), # Fetch more to allow filtering
            mode="hybrid",
            label_filter=label_filter,
        )

        if not candidates:
            return result

        # 基于相对分数做轻量置信度分层
        max_score = max(float(item.get("score", 0.0)) for item in candidates) or 1.0

        matches: list[IndexMatch] = []
        
        # Valid code extensions
        valid_code_exts = {".cs", ".lua", ".py", ".cpp", ".h", ".c", ".js", ".ts"}

        for item in candidates:
            label = str(item.get("label", ""))
            if node_types and label not in node_types:
                continue

            # Filter CodeFile by extension
            if label == "CodeFile":
                name = str(item.get("name", item.get("id", ""))).lower()
                if not any(name.endswith(ext) for ext in valid_code_exts):
                    continue

            raw_score = float(item.get("score", 0.0))
            confidence = max(0.0, min(1.0, raw_score / max_score))
            level = "high" if confidence >= 0.6 else "low"
            if high_confidence_only and level != "high":
                continue

            matches.append(
                IndexMatch(
                    term_name=str(item.get("name", item.get("id", ""))),
                    node_type=label,
                    asset_link=self._extract_asset_link(item),
                    confidence=confidence,
                    confidence_level=level,
                    match_reason="hybrid_index",
                    raw_score=raw_score,
                )
            )

            if len(matches) >= top_k:
                break

        for match in matches:
            if match.confidence_level == "high":
                result.high_confidence_matches.append(match)
            else:
                result.low_confidence_matches.append(match)

        return result


# =============================================================================
# GraphIndexOnlyTool - Agent 工具封装
# =============================================================================

class GraphIndexOnlyTool:
    """
    Index-only 模式工具

    供 Agent 调用的接口
    """

    name = "graph_index_only"
    description = (
        "Index-only 检索：只返回术语名、类型、资产链接，不返回详情。"
        "适用于快速定位术语和文件，作为探索入口。"
        "如果 include_repo_map 为 true 且查到了代码文件，会附带它们的 Repo Map。"
    )

    def __init__(
        self,
        db_path: Optional[str] = None,
        git_root: str = "record/",
        wiki_domain: str = "https://xxx.wiki.cn"
    ):
        """初始化工具

        Args:
            db_path: 图数据库路径，默认从环境变量或标准路径获取
        """
        if db_path is None:
            db_path = os.environ.get(
                "GRAPH_RAG_DB_PATH",
                "/home/developer/projects/open_graph-graph-rag/data/kg_graph.db"
            )
        self.db_path = db_path
        self.retriever = IndexOnlyRetriever(
            git_root=git_root,
            wiki_domain=wiki_domain
        )

    def to_schema(self) -> Dict[str, Any]:
        """返回工具 Schema（OpenAI Function Calling 格式）"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "查询语句 (e.g. '登录鉴权', 'BattleManager')"
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "返回结果数量",
                            "default": 10
                        },
                        "high_confidence_only": {
                            "type": "boolean",
                            "description": "只返回高确信结果",
                            "default": False
                        },
                        "node_types": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "筛选节点类型 (e.g. ['CodeFile', 'BusinessTerm'])"
                        },
                        "include_repo_map": {
                            "type": "boolean",
                            "description": "是否为找到的代码文件附加 Repo Map",
                            "default": False
                        }
                    },
                    "required": ["query"]
                }
            }
        }

    def __call__(
        self,
        query: str,
        top_k: int = 10,
        high_confidence_only: bool = False,
        node_types: Optional[List[str]] = None,
        include_repo_map: bool = False
    ) -> str:
        """执行检索

        Args:
            query: 查询语句
            top_k: 返回数量上限
            high_confidence_only: 只返回高确信结果
            node_types: 筛选的节点类型列表
            include_repo_map: 是否附加 Repo Map

        Returns:
            格式化后的检索结果
        """
        try:
            result = self.retriever.retrieve(
                query=query,
                top_k=top_k,
                high_confidence_only=high_confidence_only,
                node_types=node_types
            )
            out_str = result.format()

            if include_repo_map:
                from src.tools.repo_map import generate_repo_map
                import os

                code_files = []
                for m in result.high_confidence_matches + result.low_confidence_matches:
                    if m.node_type == "CodeFile":
                        # Convert Git link back to local path, or handle raw node name
                        # We use the asset_link since it contains the full relative path for RepoMap
                        path = m.asset_link if m.asset_link else m.term_name
                        code_files.append(path)

                if code_files:
                    out_str += "\n\n【Repo Map 附加信息】\n"
                    bridge_url = os.environ.get("WINDOWS_FILE_BRIDGE_URL", "")
                    if bridge_url:
                        for cf in code_files:
                            try:
                                # Git link is like "src/main/..."
                                # We need to pass the directory properly.
                                # Let's assume the root is the first part or just pass "." and use cf as directory.
                                rm = generate_repo_map(root_path=".", directory=cf, max_files=1, remote_bridge_url=bridge_url)
                                if rm and "No code structure found" not in rm:
                                    out_str += f"\n--- Repo Map for {cf} ---\n{rm}\n"
                            except Exception as re:
                                out_str += f"\n(Failed to generate Repo Map for {cf}: {re})\n"
                    else:
                        out_str += "\n[警告] 未配置 WINDOWS_FILE_BRIDGE_URL，无法生成远程 Repo Map。\n"

            return out_str
        except Exception as e:
            logger.error(f"Index-only 检索失败：{e}")
            return f"检索失败：{e}"


# =============================================================================
# 便捷函数
# =============================================================================

def index_only_search(
    query: str,
    top_k: int = 10,
    high_confidence_only: bool = False,
    node_types: Optional[List[str]] = None,
    include_repo_map: bool = False,
    db_path: Optional[str] = None
) -> str:
    """便捷函数：执行 index-only 检索"""
    tool = GraphIndexOnlyTool(db_path=db_path)
    return tool(query, top_k=top_k, high_confidence_only=high_confidence_only, node_types=node_types, include_repo_map=include_repo_map)


# =============================================================================
# 测试
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 测试检索
    test_queries = [
        "BattleManager",
        "登录",
        "Guild War",
        "network",
    ]

    tool = GraphIndexOnlyTool()

    print("=" * 60)
    print("Index-Only 检索测试")
    print("=" * 60)

    for query in test_queries:
        print(f"\n查询：{query}")
        print("-" * 40)
        result = tool(query, top_k=5)
        print(result)
        print()
