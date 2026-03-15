# -*- coding: utf-8 -*-
"""
代码片段向量索引 (Snippet Index)

从 code_snippets.db 流式读取 embedding，
提供 search() 方法返回 {codefile_node_id: max_snippet_score}。
"""

import logging
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class SnippetIndex:
    """流式片段向量索引（按需读盘，不做全量内存缓存）"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._loaded = False
        self._snippet_count = 0

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def snippet_count(self) -> int:
        return self._snippet_count

    def load(self):
        """加载片段索引元信息（不加载全量向量）"""
        if not Path(self.db_path).exists():
            logger.warning(f"[snippet_index] 片段库不存在: {self.db_path}")
            self._loaded = False
            self._snippet_count = 0
            return

        t0 = time.time()
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT COUNT(*) FROM snippets WHERE embedding IS NOT NULL"
        ).fetchone()
        conn.close()

        self._snippet_count = int(row[0]) if row else 0
        self._loaded = True

        elapsed = time.time() - t0
        logger.info(
            f"[snippet_index] 元信息加载完成: {self._snippet_count} 片段, 耗时 {elapsed:.2f}s"
        )

    def _iter_snippets(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute(
                "SELECT id, codefile_node_id, embedding FROM snippets WHERE embedding IS NOT NULL"
            )
            for row in cursor:
                yield row
        finally:
            conn.close()

    def search(self, query_vec: np.ndarray, top_k: int = 20) -> Dict[str, float]:
        """
        搜索片段，返回 {codefile_node_id: max_snippet_score}

        Args:
            query_vec: 归一化的查询向量 (1D)
            top_k: 返回的 codefile 数量上限

        Returns:
            codefile_node_id -> 该文件所有片段中的最高余弦相似度
        """
        if not self.is_loaded or self._snippet_count <= 0:
            return {}

        codefile_scores: Dict[str, float] = {}

        for row in self._iter_snippets():
            blob = row["embedding"]
            if not blob:
                continue
            vec = np.frombuffer(blob, dtype=np.float32)
            if len(vec) == 0:
                continue

            norm = np.linalg.norm(vec)
            if norm <= 0:
                continue
            vec = vec / norm
            score = float(np.dot(vec, query_vec))

            cf_id = row["codefile_node_id"]
            if cf_id not in codefile_scores or score > codefile_scores[cf_id]:
                codefile_scores[cf_id] = score

        sorted_items = sorted(codefile_scores.items(), key=lambda x: -x[1])
        return dict(sorted_items[:top_k])

    def get_top_snippets(
        self, query_vec: np.ndarray, codefile_node_id: str, top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """
        获取指定 codefile 中得分最高的片段 ID

        用于检索命中后展示具体匹配的代码片段。
        """
        if not self.is_loaded or self._snippet_count <= 0:
            return []

        results: List[Tuple[str, float]] = []
        for row in self._iter_snippets():
            if row["codefile_node_id"] != codefile_node_id:
                continue

            blob = row["embedding"]
            if not blob:
                continue
            vec = np.frombuffer(blob, dtype=np.float32)
            if len(vec) == 0:
                continue

            norm = np.linalg.norm(vec)
            if norm <= 0:
                continue
            vec = vec / norm
            score = float(np.dot(vec, query_vec))
            results.append((row["id"], score))

        results.sort(key=lambda x: -x[1])
        return results[:top_k]
