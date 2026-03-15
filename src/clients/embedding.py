# -*- coding: utf-8 -*-
"""
Embedding 客户端

使用 sentence-transformers 本地生成向量。
"""

import logging
from typing import List, Optional

from src.config import get_settings

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """Embedding 客户端 (单例)"""
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingClient, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._model is None:
            self._init_model()

    def _init_model(self):
        from sentence_transformers import SentenceTransformer
        from transformers.utils import logging as hf_logging
        hf_logging.disable_progress_bar()
        config = get_settings()
        model_name = "BAAI/bge-m3"
        if hasattr(config, "rag") and config.rag:
            if isinstance(config.rag, dict):
                model_name = config.rag.get("model_name", model_name)
            else:
                model_name = getattr(config.rag, "model_name", model_name)

        logger.info(f"正在加载 Embedding 模型: {model_name} ...")
        try:
            self._model = SentenceTransformer(model_name)
            logger.info("Embedding 模型加载完成")
        except Exception as e:
            logger.error(f"加载 Embedding 模型失败: {e}")
            raise

    def embed_text(self, text: str) -> List[float]:
        if not text:
            return []
        embedding = self._model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        embeddings = self._model.encode(texts, convert_to_numpy=True, batch_size=32, show_progress_bar=False)
        return embeddings.tolist()

    @property
    def vector_dimension(self) -> int:
        if self._model:
            return self._model.get_sentence_embedding_dimension()
        return 0


def get_embedding_client() -> EmbeddingClient:
    return EmbeddingClient()
