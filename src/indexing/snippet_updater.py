# -*- coding: utf-8 -*-
"""
代码片段增量更新器 (Snippet Updater)

职责:
1. 从主库读取 CodeFile 节点列表
2. 通过桥接服务读取文件内容
3. 用 code_chunker 分片
4. 用 embedding 客户端批量生成向量
5. 写入片段数据库 (code_snippets.db)
6. 基于 SHA256 hash 做增量更新

设计为后台线程运行，不阻塞主服务。
"""

import hashlib
import json
import logging
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests

from src.indexing.code_chunker import CodeChunk, chunk_code

logger = logging.getLogger(__name__)

# 当前索引版本 — 分片策略或 embedding 模型变更时递增，强制全量重建
INDEX_VERSION = 1
BATCH_SIZE = 64


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _build_embedding_text(chunk: CodeChunk, file_path: str, lang: str) -> str:
    """构建用于 embedding 的结构化文本"""
    parts = [
        f"File: {file_path}",
        f"Language: {lang}",
        f"{chunk.symbol_type.title()}: {chunk.symbol_name}",
        chunk.code[:3000],
    ]
    return "\n".join(parts)
class SnippetDB:
    """片段数据库管理"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._ensure_schema()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
        return self._conn

    def _ensure_schema(self):
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS snippets (
                id TEXT PRIMARY KEY,
                codefile_node_id TEXT NOT NULL,
                file_path TEXT NOT NULL,
                lang TEXT NOT NULL,
                symbol_type TEXT NOT NULL,
                symbol_name TEXT NOT NULL,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                line_count INTEGER NOT NULL,
                code_text TEXT NOT NULL,
                embedding BLOB,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS file_index_status (
                codefile_node_id TEXT PRIMARY KEY,
                file_path TEXT NOT NULL,
                file_size INTEGER,
                file_hash TEXT,
                snippet_count INTEGER DEFAULT 0,
                index_version INTEGER DEFAULT 1,
                indexed_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_snippets_codefile ON snippets(codefile_node_id);
            CREATE INDEX IF NOT EXISTS idx_snippets_lang ON snippets(lang);
        """)
        conn.commit()

    def get_file_status(self) -> Dict[str, dict]:
        """返回 {codefile_node_id: {file_hash, index_version, ...}}"""
        conn = self._get_conn()
        rows = conn.execute("SELECT * FROM file_index_status").fetchall()
        return {r["codefile_node_id"]: dict(r) for r in rows}

    def upsert_file_snippets(
        self,
        codefile_node_id: str,
        file_path: str,
        file_hash: str,
        file_size: int,
        snippets: List[Tuple[CodeChunk, Optional[bytes]]],
    ):
        """原子写入一个文件的所有片段"""
        conn = self._get_conn()
        now = _now_iso()

        # 删除旧片段
        conn.execute("DELETE FROM snippets WHERE codefile_node_id = ?", (codefile_node_id,))

        # 插入新片段
        for chunk, emb_blob in snippets:
            snip_id = f"snip:{codefile_node_id}:{chunk.start_line}"
            lang = Path(file_path).suffix.lstrip(".").replace("cs", "csharp")
            conn.execute(
                """INSERT INTO snippets
                   (id, codefile_node_id, file_path, lang, symbol_type, symbol_name,
                    start_line, end_line, line_count, code_text, embedding, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (snip_id, codefile_node_id, file_path, lang,
                 chunk.symbol_type, chunk.symbol_name,
                 chunk.start_line, chunk.end_line, chunk.line_count,
                 chunk.code, emb_blob, now, now),
            )

        # 更新文件状态
        conn.execute(
            """INSERT OR REPLACE INTO file_index_status
               (codefile_node_id, file_path, file_size, file_hash,
                snippet_count, index_version, indexed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (codefile_node_id, file_path, file_size, file_hash,
             len(snippets), INDEX_VERSION, now),
        )
        conn.commit()

    def total_snippets(self) -> int:
        return self._get_conn().execute("SELECT COUNT(*) FROM snippets").fetchone()[0]

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None
class SnippetUpdater:
    """
    增量更新器

    从主库读取 CodeFile → 桥接服务读文件 → 分片 → embedding → 写片段库
    """

    def __init__(
        self,
        main_db_path: str,
        snippet_db_path: str,
        bridge_url: str,
        embedder=None,
    ):
        self.main_db_path = main_db_path
        self.snippet_db = SnippetDB(snippet_db_path)
        self.bridge_url = bridge_url.rstrip("/")
        self._embedder = embedder

    def _get_embedder(self):
        if self._embedder is None:
            from src.clients.embedding import get_embedding_client
            self._embedder = get_embedding_client()
        return self._embedder

    def _get_codefile_nodes(self) -> List[dict]:
        """从主库读取所有 CodeFile 节点"""
        conn = sqlite3.connect(self.main_db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, properties FROM nodes WHERE label = 'CodeFile'"
        ).fetchall()
        conn.close()

        results = []
        for r in rows:
            props = json.loads(r["properties"]) if r["properties"] else {}
            path = props.get("path", "")
            if not path:
                continue
            # 只处理 .lua 和 .cs 文件
            ext = Path(path).suffix.lower()
            if ext not in (".lua", ".cs"):
                continue
            results.append({"id": r["id"], "path": path})
        return results

    def _read_remote_file(self, file_path: str) -> Optional[str]:
        """通过桥接服务读取文件内容"""
        # 去掉 record/ 前缀
        remote_path = file_path
        for prefix in ("record/", "Record/"):
            if remote_path.startswith(prefix):
                remote_path = remote_path[len(prefix):]
                break
        try:
            resp = requests.post(
                f"{self.bridge_url}/read_file",
                json={"path": remote_path},
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                if data.get("success"):
                    return data.get("content", "")
        except requests.RequestException as e:
            logger.debug(f"读取文件失败 {remote_path}: {e}")
        return None

    def _embed_batch(self, texts: List[str]) -> List[bytes]:
        """批量生成 embedding，返回 BLOB 列表"""
        embedder = self._get_embedder()
        vectors = embedder.embed_batch(texts)
        blobs = []
        for vec in vectors:
            arr = np.array(vec, dtype=np.float32)
            blobs.append(arr.tobytes())
        return blobs

    def run(self, max_files: int = 0) -> dict:
        """
        执行增量更新。

        Args:
            max_files: 最多处理文件数 (0=不限)

        Returns:
            统计信息 dict
        """
        t0 = time.time()
        stats = {"total_codefiles": 0, "skipped": 0, "indexed": 0,
                 "failed": 0, "snippets_created": 0}

        # 1. 读取主库 CodeFile 列表
        codefiles = self._get_codefile_nodes()
        stats["total_codefiles"] = len(codefiles)
        logger.info(f"[snippet_updater] 主库 CodeFile: {len(codefiles)} 个 (.lua/.cs)")

        # 2. 读取片段库状态
        file_status = self.snippet_db.get_file_status()

        # 3. 筛选需要更新的文件
        to_process = []
        for cf in codefiles:
            status = file_status.get(cf["id"])
            if status is None:
                to_process.append(cf)  # 新文件
            elif status.get("index_version", 0) != INDEX_VERSION:
                to_process.append(cf)  # 版本不一致
            else:
                stats["skipped"] += 1

        logger.info(
            f"[snippet_updater] 需处理: {len(to_process)}, "
            f"跳过: {stats['skipped']}"
        )

        if max_files > 0:
            to_process = to_process[:max_files]

        # 4. 分批处理
        batch_texts = []  # embedding 文本
        batch_meta = []   # (codefile, chunk, file_hash, file_size) 对应关系
        pending_files = {}  # codefile_id -> [(chunk, emb_blob_placeholder)]

        for i, cf in enumerate(to_process):
            if (i + 1) % 100 == 0 or i == 0:
                logger.info(
                    f"[snippet_updater] 进度: {i+1}/{len(to_process)}, "
                    f"已生成 {stats['snippets_created']} 片段"
                )

            # 读取文件
            content = self._read_remote_file(cf["path"])
            if not content:
                stats["failed"] += 1
                continue

            file_hash = _sha256(content)
            file_size = len(content.encode("utf-8"))

            # 检查 hash 是否变化
            status = file_status.get(cf["id"])
            if status and status.get("file_hash") == file_hash:
                stats["skipped"] += 1
                continue

            # 分片
            chunks = chunk_code(content, cf["path"])
            if not chunks:
                stats["failed"] += 1
                continue

            # 构建 embedding 文本
            lang = Path(cf["path"]).suffix.lstrip(".")
            for chunk in chunks:
                emb_text = _build_embedding_text(chunk, cf["path"], lang)
                batch_texts.append(emb_text)
                batch_meta.append((cf["id"], cf["path"], chunk, file_hash, file_size))

            # 达到批量大小时生成 embedding
            if len(batch_texts) >= BATCH_SIZE:
                self._flush_batch(batch_texts, batch_meta, stats)
                batch_texts.clear()
                batch_meta.clear()

        # 处理剩余
        if batch_texts:
            self._flush_batch(batch_texts, batch_meta, stats)

        elapsed = time.time() - t0
        stats["elapsed_seconds"] = round(elapsed, 1)
        stats["total_snippets"] = self.snippet_db.total_snippets()

        logger.info(
            f"[snippet_updater] 完成! "
            f"索引 {stats['indexed']} 文件, "
            f"生成 {stats['snippets_created']} 片段, "
            f"总计 {stats['total_snippets']} 片段, "
            f"耗时 {elapsed:.1f}s"
        )
        return stats

    def _flush_batch(self, texts: List[str], meta: list, stats: dict):
        """批量生成 embedding 并写入数据库"""
        try:
            blobs = self._embed_batch(texts)
        except Exception as e:
            logger.error(f"[snippet_updater] embedding 生成失败: {e}")
            stats["failed"] += len(set(m[0] for m in meta))
            return

        # 按 codefile 分组
        file_snippets: Dict[str, list] = {}  # codefile_id -> [(chunk, blob)]
        file_info: Dict[str, tuple] = {}     # codefile_id -> (path, hash, size)

        for (cf_id, path, chunk, fhash, fsize), blob in zip(meta, blobs):
            file_snippets.setdefault(cf_id, []).append((chunk, blob))
            file_info[cf_id] = (path, fhash, fsize)

        # 写入数据库
        for cf_id, snippets in file_snippets.items():
            path, fhash, fsize = file_info[cf_id]
            try:
                self.snippet_db.upsert_file_snippets(cf_id, path, fhash, fsize, snippets)
                stats["indexed"] += 1
                stats["snippets_created"] += len(snippets)
            except Exception as e:
                logger.error(f"[snippet_updater] 写入失败 {cf_id}: {e}")
                stats["failed"] += 1
