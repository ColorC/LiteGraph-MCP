# -*- coding: utf-8 -*-
"""
API路由定义

实现OpenAI兼容的API端点转发
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import httpx

from .config import get_config

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================================
# 请求/响应模型
# ============================================================================

class ChatMessage(BaseModel):
    """聊天消息"""
    role: str
    content: str
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    """聊天补全请求 - OpenAI兼容格式"""
    model: Optional[str] = None
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    stream: Optional[bool] = False
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = None


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    configured: bool
    model: str


# ============================================================================
# 路由处理
# ============================================================================

@router.get("/health")
async def health_check() -> HealthResponse:
    """健康检查端点"""
    config = get_config()
    return HealthResponse(
        status="ok",
        configured=config.is_configured(),
        model=config.glm_model
    )


@router.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    聊天补全端点 - 转发到GLM API
    
    完全兼容OpenAI API格式，支持：
    - 普通对话
    - 流式响应
    - Function Calling
    """
    config = get_config()
    
    if not config.is_configured():
        raise HTTPException(
            status_code=503,
            detail="API Key未配置，请设置GLM_API_KEY环境变量"
        )
    
    # 构建请求体
    payload = {
        "model": request.model or config.glm_model,
        "messages": [msg.model_dump(exclude_none=True) for msg in request.messages],
        "temperature": request.temperature,
        "stream": request.stream,
    }
    
    if request.max_tokens:
        payload["max_tokens"] = request.max_tokens
    if request.top_p:
        payload["top_p"] = request.top_p
    if request.tools:
        payload["tools"] = request.tools
    if request.tool_choice:
        payload["tool_choice"] = request.tool_choice
    
    headers = {
        "Authorization": f"Bearer {config.glm_api_key}",
        "Content-Type": "application/json"
    }
    
    target_url = f"{config.glm_base_url}/chat/completions"
    
    logger.info(f"转发请求到: {target_url}, model: {payload['model']}")
    
    try:
        if request.stream:
            # 流式响应
            return await _stream_response(target_url, headers, payload, config)
        else:
            # 普通响应
            return await _normal_response(target_url, headers, payload, config)
            
    except httpx.TimeoutException:
        logger.error("请求超时")
        raise HTTPException(status_code=504, detail="请求超时")
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP错误: {e.response.status_code} - {e.response.text}")
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"上游API错误: {e.response.text}"
        )
    except Exception as e:
        logger.error(f"请求失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _normal_response(
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    config: Any
) -> Dict[str, Any]:
    """处理普通响应"""
    async with httpx.AsyncClient(timeout=config.request_timeout) as client:
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()


async def _stream_response(
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    config: Any
) -> StreamingResponse:
    """处理流式响应"""
    async def generate():
        async with httpx.AsyncClient(timeout=config.request_timeout) as client:
            async with client.stream("POST", url, headers=headers, json=payload) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes():
                    yield chunk
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )


@router.get("/v1/models")
async def list_models():
    """列出可用模型"""
    config = get_config()
    return {
        "object": "list",
        "data": [
            {
                "id": config.glm_model,
                "object": "model",
                "created": 1700000000,
                "owned_by": "zhipu"
            }
        ]
    }
