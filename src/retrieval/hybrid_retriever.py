# -*- coding: utf-8 -*-
"""
Graph RAG 检索器 (Hybrid) - 秒级检索优化版

提供:
1. 向量相似度检索 (Semantic Search)
2. BM25 关键词检索 (Sparse Search)
3. RRF 融合 (Reciprocal Rank Fusion)
4. 术语精确匹配 (Term Lookup)
5. PPR 图扩展
"""

import heapq
import logging
import math
import pickle
from collections import Counter
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import jieba
import numpy as np

from src.clients.embedding import get_embedding_client
from src.graph.manager import LightweightGraphManager
from src.indexing.snippet_index import SnippetIndex

logger = logging.getLogger(__name__)


class GraphRAGRetriever:
    """Graph RAG 检索器（纯内存向量矩阵+倒排索引，秒级检索）"""

    def __init__(self, graph_manager: LightweightGraphManager, snippet_db_path: str = ""):
        self.gm = graph_manager
        self.embedder = get_embedding_client()

        self._term_index: Dict[str, str] = {}
        
        # Core indexes
        self._ids: List[str] = []
        self._id_to_idx: Dict[str, int] = {}
        self._is_bt_mask: Optional[np.ndarray] = None
        self._dense_matrix: Optional[np.ndarray] = None
        self._inv_index: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        # BM25 参数
        self._k1 = 1.5
        self._b = 0.75
        self._epsilon = 0.25

        # 片段向量索引
        self._snippet_index: Optional[SnippetIndex] = None
        if snippet_db_path:
            self._init_snippet_index(snippet_db_path)

        self._build_index()

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """中文分词，过滤单字符停用词"""
        return [w for w in jieba.cut(text) if len(w.strip()) > 1]

    def _init_snippet_index(self, db_path: str):
        try:
            self._snippet_index = SnippetIndex(db_path)
            self._snippet_index.load()
            if self._snippet_index.is_loaded:
                logger.info(f"Snippet index ready: {self._snippet_index.snippet_count} snippets.")
            else:
                logger.info("Snippet index: no data yet.")
        except Exception as e:
            logger.warning(f"Failed to init snippet index: {e}")
            self._snippet_index = None

    def reload_snippet_index(self):
        if self._snippet_index:
            self._snippet_index.load()
            logger.info(f"Snippet index reloaded: {self._snippet_index.snippet_count} snippets.")

    def _build_index(self):
        """构建/加载高速缓存矩阵和倒排索引"""
        cache_path = self.gm.db_path.parent / "retriever_cache_v2.pkl"
        
        if cache_path.exists():
            logger.info(f"Loading high-speed index cache from {cache_path}...")
            try:
                with open(cache_path, "rb") as f:
                    cache = pickle.load(f)
                self._ids = cache["ids"]
                self._id_to_idx = cache["id_to_idx"]
                self._is_bt_mask = cache["is_bt_mask"]
                self._dense_matrix = cache["dense_matrix"]
                self._inv_index = cache["inv_index"]
                self._term_index = cache["term_index"]
                logger.info(f"Loaded {len(self._ids)} nodes into memory index.")
                return
            except Exception as e:
                logger.warning(f"Cache load failed: {e}. Rebuilding...")

        logger.info("Building hybrid high-speed index from graph...")

        global_df = Counter()
        global_doc_lens = []
        all_tokens = []
        vecs = []

        for node in self.gm.iter_all_nodes():
            nid = node["id"]
            is_bt = node.get("label") == "BusinessTerm"
            name = node.get("name", "")

            self._id_to_idx[nid] = len(self._ids)
            self._ids.append(nid)

            # 1. 稠密向量
            vec = node.get("_embedding")
            if vec and isinstance(vec, list) and len(vec) > 0:
                nvec = self._normalize_vec(vec)
                vecs.append(nvec if nvec is not None else np.zeros(1024, dtype=np.float32))
            else:
                vecs.append(np.zeros(1024, dtype=np.float32))

            # 2. 稀疏文本 (BM25)
            text_parts = []
            if name: text_parts.append(name)
            desc = node.get("description", "")
            if desc: text_parts.append(desc)
            if is_bt:
                aliases = node.get("discovered_aliases") or node.get("aliases") or []
                text_parts.extend([str(a) for a in aliases if a])
                summary = node.get("summary", "")
                if summary: text_parts.append(summary)

            if text_parts:
                tokens = self._tokenize(" ".join(text_parts))
            else:
                tokens = []

            all_tokens.append(tokens)
            global_doc_lens.append(len(tokens))
            for t in set(tokens):
                global_df[t] += 1

            if is_bt and name:
                self._term_index[name.lower()] = nid

        corpus_size = len(self._ids)
        if corpus_size == 0:
            logger.warning("Graph is empty!")
            return

        self._dense_matrix = np.vstack(vecs).astype(np.float32)
        
        bt_mask = [1.0 if self.gm.get_node(nid).get("label") == "BusinessTerm" else 0.0 for nid in self._ids]
        self._is_bt_mask = np.array(bt_mask, dtype=np.float32)

        # 构建 BM25 倒排索引
        avgdl = sum(global_doc_lens) / corpus_size if corpus_size > 0 else 1.0
        idf = {}
        idf_sum = 0.0
        negative_terms = []
        for word, freq in global_df.items():
            score = math.log(corpus_size - freq + 0.5) - math.log(freq + 0.5)
            idf[word] = score
            idf_sum += score
            if score < 0:
                negative_terms.append(word)
                
        average_idf = idf_sum / len(idf) if idf else 0.0
        eps = self._epsilon * average_idf
        for w in negative_terms:
            idf[w] = eps

        inv_index_temp = {}
        for i, tokens in enumerate(all_tokens):
            tf = Counter(tokens)
            doc_len = global_doc_lens[i]
            for q, q_freq in tf.items():
                if q not in inv_index_temp:
                    inv_index_temp[q] = ([], [])
                idf_val = idf.get(q, 0.0)
                denom = q_freq + self._k1 * (1 - self._b + self._b * doc_len / avgdl)
                val = idf_val * (q_freq * (self._k1 + 1) / denom)
                inv_index_temp[q][0].append(i)
                inv_index_temp[q][1].append(val)

        for q in inv_index_temp:
            self._inv_index[q] = (
                np.array(inv_index_temp[q][0], dtype=np.int32),
                np.array(inv_index_temp[q][1], dtype=np.float32)
            )

        logger.info("Caching index to disk...")
        try:
            with open(cache_path, "wb") as f:
                pickle.dump({
                    "ids": self._ids,
                    "id_to_idx": self._id_to_idx,
                    "is_bt_mask": self._is_bt_mask,
                    "dense_matrix": self._dense_matrix,
                    "inv_index": self._inv_index,
                    "term_index": self._term_index
                }, f)
        except Exception as e:
            logger.warning(f"Could not save cache: {e}")

        logger.info(f"Metadata built: total_nodes={corpus_size}, terms={len(self._term_index)}")

    @staticmethod
    def _normalize_vec(vec: List[float]) -> Optional[np.ndarray]:
        if not vec:
            return None
        arr = np.array(vec, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm <= 0:
            return None
        return arr / norm

    def _dense_retrieve(self, query: str, top_k: int, bt_only: bool = False) -> List[Tuple[str, float]]:
        if self._dense_matrix is None:
            return []
        query_vec = self.embedder.embed_text(query)
        qv = self._normalize_vec(query_vec)
        if qv is None:
            return []

        scores = np.dot(self._dense_matrix, qv)
        
        if not bt_only and self._snippet_index and self._snippet_index.is_loaded:
            snippet_scores = self._snippet_index.search(qv, top_k=top_k)
            for cf_id, snip_score in snippet_scores.items():
                if cf_id in self._id_to_idx:
                    idx = self._id_to_idx[cf_id]
                    scores[idx] = max(scores[idx], float(snip_score))

        if bt_only:
            scores = scores * self._is_bt_mask

        top_indices = np.argsort(scores)[-top_k:][::-1]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((self._ids[idx], float(scores[idx])))
        return results

    def _bm25_retrieve(self, query: str, top_k: int, bt_only: bool = False) -> List[Tuple[str, float]]:
        if not self._inv_index:
            return []
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores = np.zeros(len(self._ids), dtype=np.float32)
        for q in query_tokens:
            if q in self._inv_index:
                indices, values = self._inv_index[q]
                scores[indices] += values

        if bt_only:
            scores = scores * self._is_bt_mask

        top_indices = np.argsort(scores)[-top_k:][::-1]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((self._ids[idx], float(scores[idx])))
        return results

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        mode: str = "hybrid",
        rerank: bool = False,
        label_filter: str = "",
    ) -> List[Dict[str, Any]]:
        use_bt_index = label_filter == "BusinessTerm"
        fetch_k = top_k * (4 if rerank else 2)

        if mode == "dense":
            ranked = self._dense_retrieve(query, fetch_k, bt_only=use_bt_index)
        elif mode == "bm25":
            ranked = self._bm25_retrieve(query, fetch_k, bt_only=use_bt_index)
        else:
            dense_results = self._dense_retrieve(query, fetch_k, bt_only=use_bt_index)
            bm25_results = self._bm25_retrieve(query, fetch_k, bt_only=use_bt_index)
            ranked = self._rrf_fuse(dense_results, bm25_results, fetch_k)

        results = []
        for nid, score in ranked:
            node_data = self.gm.get_node(nid)
            if not node_data:
                continue
            if label_filter and not use_bt_index and node_data.get("label") != label_filter:
                continue
            entry = {
                "id": nid,
                "score": score,
                "label": node_data.get("label", ""),
                "name": node_data.get("name", nid),
                "description": node_data.get("description", ""),
            }
            if node_data.get("label") == "BusinessTerm":
                entry["aliases"] = node_data.get("discovered_aliases") or node_data.get("aliases") or []
                entry["is_broad_concept"] = node_data.get("is_broad_concept", False)
                entry["summary"] = node_data.get("summary", "")
            results.append(entry)
            if len(results) >= top_k:
                break

        return results

    def expand_with_ppr(self, seed_results: List[Dict[str, Any]], alpha: float = 0.85, top_k: int = 20) -> List[Dict[str, Any]]:
        if not seed_results:
            return []

        seed_ids = [r["id"] for r in seed_results]
        ppr_scores = self.gm.compute_ppr(seed_ids, alpha=alpha, top_k=top_k * 4)

        seen_seeds = set(seed_ids)
        expanded = []

        for nid, score in ppr_scores:
            if nid in seen_seeds:
                continue
            node_data = self.gm.get_node(nid)
            if not node_data:
                continue
            if node_data.get("infra_layer") == "L1":
                continue
            expanded.append({
                "id": nid,
                "score": score,
                "label": node_data.get("label", ""),
                "name": node_data.get("name", nid),
                "description": node_data.get("description", ""),
                "algorithm": "PPR"
            })

        return expanded[:top_k]

    def exact_match_terms(self, text: str) -> List[Dict[str, Any]]:
        found_terms = []
        text_lower = text.lower()
        for term_name, nid in self._term_index.items():
            if term_name in text_lower:
                node_data = self.gm.get_node(nid)
                if node_data:
                    found_terms.append({
                        "id": nid,
                        "name": node_data.get("name", term_name),
                        "description": node_data.get("description", ""),
                        "label": "BusinessTerm"
                    })
        return found_terms

    def format_context(self, results: List[Dict[str, Any]]) -> str:
        if not results:
            return "无相关结果。"
        lines = []
        for i, item in enumerate(results):
            lines.append(f"{i+1}. [{item['label']}] {item['name']} ({item.get('score', 0):.2f})")
            if item['description']:
                lines.append(f"   {item['description']}")
        return "\n".join(lines)

    def deep_retrieve(
        self,
        query: str,
        top_k: int = 5,
        mode: str = "hybrid",
        label_filter: str = "",
    ) -> Dict[str, Any]:
        seeds = self.retrieve(query, top_k=top_k, mode=mode, label_filter=label_filter)
        if not seeds:
            return {"seeds": [], "neighbors": [], "ppr_expanded": [], "paths": [], "narrative": "未找到相关节点。"}

        seed_ids = [s["id"] for s in seeds]

        neighbor_map = {}
        edge_list = []
        seen_edges = set()
        per_seed_nb = max(3, 15 // len(seeds))

        for seed in seeds:
            nb = self.gm.get_neighbors(seed["id"], max_hops=1, max_size=per_seed_nb * 2)
            for n in nb.get("nodes", []):
                if n["id"] not in seed_ids and n["id"] not in neighbor_map:
                    n.pop("_embedding", None)
                    neighbor_map[n["id"]] = {
                        "id": n["id"], "label": n.get("label", ""),
                        "name": n.get("name", n["id"]),
                        "description": n.get("description", "")[:150],
                    }
            for e in nb.get("edges", []):
                key = (e.get("source"), e.get("target"), e.get("relationship"))
                if key not in seen_edges:
                    seen_edges.add(key)
                    edge_list.append(e)

        neighbors = list(neighbor_map.values())[:20]

        ppr_expanded = self.expand_with_ppr(seeds, alpha=0.85, top_k=10)
        known_ids = set(seed_ids) | set(neighbor_map.keys())
        ppr_expanded = [p for p in ppr_expanded if p["id"] not in known_ids][:8]

        paths = []
        if len(seed_ids) >= 2:
            for i in range(min(len(seed_ids) - 1, 3)):
                path_result = self.gm.shortest_path_detail(seed_ids[i], seed_ids[i + 1], max_depth=4)
                if path_result.get("found"):
                    paths.append(path_result)

        narrative = self._build_narrative(seeds, neighbors, edge_list, ppr_expanded, paths)

        return {
            "seeds": seeds,
            "neighbors": neighbors,
            "ppr_expanded": ppr_expanded,
            "paths": paths,
            "narrative": narrative,
        }

    def _build_narrative(
        self,
        seeds: List[Dict],
        neighbors: List[Dict],
        edges: List[Dict],
        ppr_expanded: List[Dict],
        paths: List[Dict],
    ) -> str:
        lines = []

        lines.append("## 直接匹配节点")
        for s in seeds:
            line = f"- [{s['label']}] {s['name']} (id: {s['id']}, score: {s.get('score', 0):.3f})"
            if s.get("description"):
                line += f"\n  {s['description'][:200]}"
            if s.get("summary"):
                line += f"\n  摘要: {s['summary'][:200]}"
            lines.append(line)

        if edges:
            lines.append("\n## 关键关系")
            seed_ids = set(s["id"] for s in seeds)
            seed_edges = [e for e in edges if e.get("source") in seed_ids or e.get("target") in seed_ids]
            for e in seed_edges[:15]:
                src_name = self._id_to_name(e["source"], seeds, neighbors)
                tgt_name = self._id_to_name(e["target"], seeds, neighbors)
                lines.append(f"- {src_name} --[{e['relationship']}]--> {tgt_name}")

        if paths:
            lines.append("\n## 节点间路径")
            for p in paths:
                path_parts = []
                for i, node in enumerate(p["path"]):
                    path_parts.append(f"[{node['label']}]{node['name']}")
                    if i < len(p["edges"]):
                        path_parts.append(f"-[{p['edges'][i]['relationship']}]->")
                lines.append(f"- ({p['hops']}跳) " + " ".join(path_parts))

        if ppr_expanded:
            lines.append("\n## 图结构关联发现")
            for p in ppr_expanded:
                lines.append(f"- [{p['label']}] {p['name']} (PPR: {p.get('score', 0):.4f})")

        if neighbors:
            lines.append("\n## 邻域上下文")
            by_label = {}
            for n in neighbors:
                by_label.setdefault(n["label"], []).append(n)
            for label, ns in by_label.items():
                names = ", ".join(n["name"] for n in ns[:5])
                if len(ns) > 5:
                    names += f" 等{len(ns)}个"
                lines.append(f"- {label}: {names}")

        return "\n".join(lines)

    @staticmethod
    def _id_to_name(node_id: str, seeds: List[Dict], neighbors: List[Dict]) -> str:
        for s in seeds:
            if s["id"] == node_id:
                return s["name"]
        for n in neighbors:
            if n["id"] == node_id:
                return n["name"]
        if ":" in node_id:
            return node_id.split(":", 1)[1]
        return node_id

    @staticmethod
    def _rrf_fuse(
        dense_results: List[Tuple[str, float]],
        bm25_results: List[Tuple[str, float]],
        top_k: int,
        k: int = 60,
    ) -> List[Tuple[str, float]]:
        rrf_scores: Dict[str, float] = {}
        for rank, (nid, _) in enumerate(dense_results, 1):
            rrf_scores[nid] = rrf_scores.get(nid, 0.0) + 1.0 / (k + rank)
        for rank, (nid, _) in enumerate(bm25_results, 1):
            rrf_scores[nid] = rrf_scores.get(nid, 0.0) + 1.0 / (k + rank)
        sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:top_k]


_retriever_instance = None

def get_graph_retriever() -> GraphRAGRetriever:
    global _retriever_instance
    if _retriever_instance is None:
        from src.config import get_settings
        settings = get_settings()
        data_dir = Path(settings.paths.data_dir)
        db_path = str(data_dir / "kg_graph.db")
        snippet_db_path = str(data_dir / "code_snippets.db")
        gm = LightweightGraphManager(db_path)
        _retriever_instance = GraphRAGRetriever(gm, snippet_db_path=snippet_db_path)
    return _retriever_instance
