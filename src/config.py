# -*- coding: utf-8 -*-
"""
OpenGraph GraphRAG MCP 服务器配置

支持:
1. YAML配置文件 (config/default.yaml)
2. 环境变量覆盖
3. Pydantic类型验证

配置优先级: 环境变量 > yaml > 默认值
"""

import os
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field
import yaml

logger = logging.getLogger(__name__)

# 项目根目录 (extensions/open_graph-graph-rag/)
PROJECT_ROOT = Path(__file__).parent.parent


class LLMConfig(BaseModel):
    """LLM配置"""
    use_proxy: bool = Field(default=False)
    proxy_url: str = Field(default="http://localhost:8000/v1")
    base_url: str = Field(default="https://open.bigmodel.cn/api/coding/paas/v4")
    api_key: str = Field(default="")
    model: str = Field(default="glm-5")
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=4096, ge=1)
    timeout: int = Field(default=600)

    def get_effective_url(self) -> str:
        if self.use_proxy:
            return self.proxy_url
        return self.base_url

    def get_effective_key(self) -> str:
        if self.use_proxy:
            return "proxy"
        return self.api_key


class RAGConfig(BaseModel):
    """RAG/Embedding配置"""
    model_name: str = Field(default="BAAI/bge-m3")
    model_cache_dir: Optional[str] = Field(default=None)


class WikiConfig(BaseModel):
    """Wiki配置"""
    app_id: str = Field(default="")
    app_secret: str = Field(default="")
    wiki_space_id: str = Field(default="")
    mcp_read_url: str = Field(default="")
    mcp_write_url: str = Field(default="")


class PathConfig(BaseModel):
    """路径配置"""
    git_main_root: str = Field(default=r"D:\Git\main")
    git_dev_root: str = Field(default=r"D:\Git\dev")
    data_dir: str = Field(default="data")
    logs_dir: str = Field(default="logs")


class Settings(BaseModel):
    """全局配置"""
    llm: LLMConfig = Field(default_factory=LLMConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    wiki: WikiConfig = Field(default_factory=WikiConfig)
    paths: PathConfig = Field(default_factory=PathConfig)

    root_path: Path = Field(default=PROJECT_ROOT)
    log_level: str = Field(default="INFO")

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "Settings":
        config_dict: Dict[str, Any] = {}

        if config_path is None:
            config_path = PROJECT_ROOT / "config" / "default.yaml"

        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    yaml_config = yaml.safe_load(f) or {}
                    config_dict.update(yaml_config)
                logger.info(f"已加载配置文件: {config_path}")
            except Exception as e:
                logger.warning(f"配置文件加载失败: {e}")

        env_overrides = cls._load_env_overrides()
        config_dict = cls._deep_merge(config_dict, env_overrides)

        return cls(**config_dict)

    @classmethod
    def _load_env_overrides(cls) -> Dict[str, Any]:
        overrides: Dict[str, Any] = {}

        llm_overrides = {}
        if os.getenv("LLM_USE_PROXY"):
            llm_overrides["use_proxy"] = os.getenv("LLM_USE_PROXY").lower() == "true"
        if os.getenv("LLM_PROXY_URL"):
            llm_overrides["proxy_url"] = os.getenv("LLM_PROXY_URL")
        if os.getenv("LLM_BASE_URL"):
            llm_overrides["base_url"] = os.getenv("LLM_BASE_URL")
        if os.getenv("LLM_API_KEY") or os.getenv("GLM_API_KEY"):
            llm_overrides["api_key"] = os.getenv("LLM_API_KEY") or os.getenv("GLM_API_KEY")
        if os.getenv("LLM_MODEL"):
            llm_overrides["model"] = os.getenv("LLM_MODEL")
        if llm_overrides:
            overrides["llm"] = llm_overrides

        wiki_overrides = {}
        if os.getenv("WIKI_APP_ID"):
            wiki_overrides["app_id"] = os.getenv("WIKI_APP_ID")
        if os.getenv("WIKI_APP_SECRET"):
            wiki_overrides["app_secret"] = os.getenv("WIKI_APP_SECRET")
        if wiki_overrides:
            overrides["wiki"] = wiki_overrides

        if os.getenv("LOG_LEVEL"):
            overrides["log_level"] = os.getenv("LOG_LEVEL")

        # 数据目录覆盖
        if os.getenv("open_graph_DATA_DIR"):
            overrides.setdefault("paths", {})["data_dir"] = os.getenv("open_graph_DATA_DIR")

        return overrides

    @staticmethod
    def _deep_merge(base: Dict, override: Dict) -> Dict:
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = Settings._deep_merge(result[key], value)
            else:
                result[key] = value
        return result


# 全局配置实例
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings.load()
    return _settings


def reload_settings(config_path: Optional[Path] = None) -> Settings:
    global _settings
    _settings = Settings.load(config_path)
    return _settings
