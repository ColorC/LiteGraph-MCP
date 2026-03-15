# -*- coding: utf-8 -*-
"""
后端代理服务配置

从环境变量读取敏感配置，确保API Key安全
"""

import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field


class ServerConfig(BaseModel):
    """服务端配置"""
    
    # GLM API配置
    glm_api_key: str = Field(
        default="",
        description="GLM API Key，从环境变量GLM_API_KEY读取"
    )
    glm_base_url: str = Field(
        default="https://open.bigmodel.cn/api/coding/paas/v4",
        description="GLM API基础URL"
    )
    glm_model: str = Field(
        default="glm-5",
        description="默认使用的GLM模型"
    )
    
    # 服务配置
    host: str = Field(default="0.0.0.0", description="监听地址")
    port: int = Field(default=8000, description="监听端口")
    
    # 超时配置
    request_timeout: int = Field(default=120, description="请求超时时间(秒)")
    
    # 日志配置
    log_level: str = Field(default="INFO", description="日志级别")
    
    # 图数据库配置
    graph_db_path: str = Field(
        default="data/knowledge/graph.db",
        description="知识图谱数据库路径"
    )
    
    @classmethod
    def from_env(cls) -> "ServerConfig":
        """从环境变量加载配置"""
        return cls(
            glm_api_key=os.getenv("GLM_API_KEY", ""),
            glm_base_url=os.getenv(
                "GLM_BASE_URL",
                "https://open.bigmodel.cn/api/coding/paas/v4"
            ),
            glm_model=os.getenv("GLM_MODEL", "glm-5"),
            host=os.getenv("SERVER_HOST", "0.0.0.0"),
            port=int(os.getenv("SERVER_PORT", "8000")),
            request_timeout=int(os.getenv("REQUEST_TIMEOUT", "120")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            graph_db_path=os.getenv(
                "GRAPH_DB_PATH", 
                str(Path(__file__).resolve().parents[2] / "data" / "kg_graph.db")
            ),
        )
    
    def is_configured(self) -> bool:
        """检查是否已配置API Key"""
        return bool(self.glm_api_key)


# 全局配置实例
_config: Optional[ServerConfig] = None


def get_config() -> ServerConfig:
    """获取服务配置"""
    global _config
    if _config is None:
        _config = ServerConfig.from_env()
    return _config
