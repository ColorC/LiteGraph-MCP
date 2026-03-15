# -*- coding: utf-8 -*-
"""
Wiki MCP 客户端

通过 HTTP 直接调用Wiki托管的 MCP 服务，无需启动本地进程。

支持的服务:
- wiki_wiki: Wiki知识库只读 (搜索、阅读文档)
- wiki_project: Wiki项目管理 (需求/Story 查询) — 需要 OAuth

调用协议: JSON-RPC 2.0 over HTTP
"""

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


class WikiMCPClient:
    """
    Wiki MCP 客户端

    通过 HTTP 直接调用Wiki托管的 MCP 服务。
    """

    def __init__(
        self,
        wiki_url: str = "https://open.wiki.cn/mcp/stream/mcp_zohhcjEYgBomUF1c8lLwwQD7IOpvAzuSxRniVoF6y7mR0dBTq-12J50jKSLFrFNxu0aEYPaJ2uo",
        project_url: str = "https://project.wiki.cn/mcp_server/v1",
        wiki_space_id: str = "7065713946103119900",
        timeout: int = 60,
    ):
        self.wiki_url = wiki_url
        self.project_url = project_url
        self.wiki_space_id = wiki_space_id
        self.timeout = timeout

        self._client: Optional[httpx.AsyncClient] = None
        self._project_token: Optional[str] = None
        self._wiki_available: Optional[bool] = None  # None=未检测, True/False
        self._project_available: Optional[bool] = None
        self._request_id = 0

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    def _next_request_id(self) -> int:
        self._request_id += 1
        return self._request_id

    async def _call_mcp(
        self,
        url: str,
        method: str,
        params: Dict[str, Any] = None,
        headers: Dict[str, str] = None,
    ) -> Dict[str, Any]:
        """调用 MCP 服务 (JSON-RPC 2.0)"""
        client = await self._get_client()

        req_id = self._next_request_id()
        payload = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params or {},
        }

        req_headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if headers:
            req_headers.update(headers)

        try:
            response = await client.post(url, json=payload, headers=req_headers)
            response.raise_for_status()

            # 处理 SSE 或 JSON 响应
            content_type = response.headers.get("content-type", "")
            if "text/event-stream" in content_type:
                # SSE 响应: 解析最后一个 data 块
                result = self._parse_sse_response(response.text)
            else:
                result = response.json()

            if "error" in result:
                raise MCPError(result["error"])

            return result.get("result", result)

        except httpx.HTTPStatusError as e:
            logger.error(f"MCP HTTP error: {e.response.status_code} - {e.response.text}")
            raise MCPError(f"HTTP {e.response.status_code}: {e.response.text}")
        except Exception as e:
            logger.error(f"MCP call failed: {e}")
            raise

    def _parse_sse_response(self, text: str) -> Dict[str, Any]:
        """解析 SSE 响应，提取最后一个 data 块"""
        lines = text.strip().split("\n")
        last_data = None
        for line in lines:
            if line.startswith("data:"):
                last_data = line[5:].strip()
        if last_data:
            try:
                return json.loads(last_data)
            except json.JSONDecodeError:
                pass
        return {"result": text}

    # =========================================================================
    # Wiki 工具
    # =========================================================================

    async def wiki_list_tools(self) -> List[Dict[str, Any]]:
        """列出Wiki Wiki MCP 的所有工具"""
        result = await self._call_mcp(self.wiki_url, "tools/list")
        return result.get("tools", [])

    async def wiki_search(self, query: str, page_size: int = 10) -> List[Dict[str, Any]]:
        """搜索Wiki知识库文档"""
        if self._wiki_available is False:
            return []  # 已知不可用，跳过网络请求
        try:
            result = await self._call_mcp(
                self.wiki_url,
                "tools/call",
                {
                    "name": "wiki_v1_node_search",
                    "arguments": {
                        "body": {"query": query, "page_size": page_size}
                    }
                }
            )
            self._wiki_available = True
            return self._extract_search_results(result)
        except MCPError as e:
            if "401" in str(e) or "403" in str(e):
                if self._wiki_available is None:
                    logger.warning("Wiki Wiki MCP 认证失败（URL 可能已过期），Wiki 工具不可用")
                    self._wiki_available = False
                return []
            raise

    async def wiki_get_node(self, token: str) -> Dict[str, Any]:
        """获取Wiki知识库节点信息"""
        if self._wiki_available is False:
            return {}
        try:
            result = await self._call_mcp(
                self.wiki_url,
                "tools/call",
                {
                    "name": "wiki_v2_space_getNode",
                    "arguments": {
                        "path": {"token": token}
                    }
                }
            )
            return result
        except MCPError as e:
            if "401" in str(e) or "403" in str(e):
                self._wiki_available = False
                return {}
            raise

    async def wiki_read_doc(self, document_id: str) -> str:
        """读取Wiki文档内容"""
        if self._wiki_available is False:
            return ""
        try:
            result = await self._call_mcp(
                self.wiki_url,
                "tools/call",
                {
                    "name": "docx_v1_documentBlock_list",
                    "arguments": {
                        "path": {"document_id": document_id}
                    }
                }
            )
            return self._extract_doc_content(result)
        except MCPError as e:
            if "401" in str(e) or "403" in str(e):
                self._wiki_available = False
                return ""
            raise

    async def wiki_list_children(self, parent_token: str, page_size: int = 50) -> List[Dict[str, Any]]:
        """列出知识库节点的子节点"""
        if self._wiki_available is False:
            return []
        try:
            result = await self._call_mcp(
                self.wiki_url,
                "tools/call",
                {
                    "name": "wiki_v2_space_listChildren",
                    "arguments": {
                        "path": {"space_id": self.wiki_space_id, "parent_token": parent_token},
                        "query": {"page_size": page_size}
                    }
                }
            )
            return self._extract_children(result)
        except MCPError as e:
            if "401" in str(e) or "403" in str(e):
                self._wiki_available = False
                return []
            raise

    def _extract_mcp_text(self, result: Dict) -> Any:
        """从 MCP 响应中提取实际数据（处理 content 数组格式）"""
        content = result.get("content", result)

        # MCP 格式: {"content": [{"type": "text", "text": "{...json...}"}]}
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text", "")
                    try:
                        return json.loads(text)
                    except json.JSONDecodeError:
                        return text
            return content

        if isinstance(content, str):
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return content

        return content

    def _extract_search_results(self, result: Dict) -> List[Dict[str, Any]]:
        """从搜索结果中提取文档列表"""
        try:
            content = self._extract_mcp_text(result)

            if isinstance(content, dict):
                inner = content.get("data", content)
                if isinstance(inner, dict):
                    inner = inner.get("data", inner)
                items = inner.get("items", [])
            else:
                items = []

            return items
        except Exception as e:
            logger.warning(f"解析搜索结果失败: {e}")
            return []

    def _extract_doc_content(self, result: Dict) -> str:
        """从文档读取结果中提取纯文本"""
        try:
            content = self._extract_mcp_text(result)

            # 解析 block 结构
            if isinstance(content, dict):
                inner = content.get("data", content)
                if isinstance(inner, dict) and "data" in inner:
                    inner = inner["data"]
                items = inner.get("items", [])
            else:
                items = []

            if not items:
                return str(content)[:5000]

            # 转换为 Markdown
            lines = []
            block_type_map = {
                1: "", 2: "", 3: "# ", 4: "## ", 5: "### ", 6: "#### ",
                12: "- ", 13: "1. ", 14: "> ", 15: "```\n",
            }

            for item in items:
                bt = item.get("block_type", 0)
                prefix = block_type_map.get(bt, "")

                # 提取文本
                elements = None
                for key in ("page", "heading1", "heading2", "heading3", "heading4",
                           "text", "bullet", "ordered", "quote", "code"):
                    block = item.get(key)
                    if block and isinstance(block, dict):
                        elements = block.get("elements", [])
                        break

                if not elements:
                    continue

                text_parts = []
                for el in elements:
                    tr = el.get("text_run", {})
                    text = tr.get("content", "")
                    if text:
                        text_parts.append(text)

                line = "".join(text_parts).strip()
                if line:
                    lines.append(f"{prefix}{line}")

            return "\n".join(lines)[:8000]
        except Exception as e:
            logger.warning(f"解析文档内容失败: {e}")
            return str(result)[:5000]

    def _extract_children(self, result: Dict) -> List[Dict[str, Any]]:
        """从子节点列表结果中提取节点数组"""
        try:
            content = self._extract_mcp_text(result)

            if isinstance(content, dict):
                inner = content.get("data", content)
                if isinstance(inner, dict):
                    inner = inner.get("data", inner)
                return inner.get("items", [])
            return []
        except Exception as e:
            logger.warning(f"解析子节点列表失败: {e}")
            return []

    # =========================================================================
    # Project 工具 (需要 OAuth)
    # =========================================================================

    async def _ensure_project_auth(self):
        """确保 Project MCP 有 OAuth token"""
        if self._project_token:
            return
        if self._project_available is False:
            return

        # 尝试从多个位置加载 token
        token_paths = [
            Path(__file__).parent.parent.parent / "data" / "wiki_oauth_token.json",
            Path.home() / ".open_graph_agent" / "wiki_user_token.json",
        ]
        for token_path in token_paths:
            if token_path.exists():
                try:
                    with open(token_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    token = data.get("access_token")
                    expires_at = data.get("expires_at", 0)
                    if token and time.time() < expires_at:
                        self._project_token = token
                        logger.info(f"从缓存加载Wiki Project OAuth token: {token_path.name}")
                        return
                except Exception as e:
                    logger.warning(f"加载 token 失败 ({token_path}): {e}")

        logger.warning("Wiki Project OAuth token 未配置或已过期，Project 工具不可用")
        logger.info("请运行: python scripts/wiki_oauth_refresh.py")
        self._project_available = False

    async def project_search_stories(self, moql: str) -> List[Dict[str, Any]]:
        """通过 MOQL 搜索Wiki项目需求"""
        await self._ensure_project_auth()
        if self._project_available is False:
            return []

        headers = {}
        if self._project_token:
            headers["Authorization"] = f"Bearer {self._project_token}"

        try:
            result = await self._call_mcp(
                self.project_url,
                "tools/call",
                {
                    "name": "search_by_mql",
                    "arguments": {"mql": moql}
                },
                headers=headers,
            )
            return result.get("items", [])
        except MCPError as e:
            if "401" in str(e) or "403" in str(e):
                logger.warning(f"Wiki Project OAuth 认证失败: {e}")
                self._project_available = False
                self._project_token = None
                return []
            raise

    async def project_get_story(self, story_id: str) -> Dict[str, Any]:
        """获取单个需求的详情"""
        await self._ensure_project_auth()
        if self._project_available is False:
            return {}

        headers = {}
        if self._project_token:
            headers["Authorization"] = f"Bearer {self._project_token}"

        try:
            result = await self._call_mcp(
                self.project_url,
                "tools/call",
                {
                    "name": "get_workitem",
                    "arguments": {"workitem_id": story_id}
                },
                headers=headers,
            )
            return result
        except MCPError as e:
            if "401" in str(e) or "403" in str(e):
                self._project_available = False
                self._project_token = None
                return {}
            raise

    # =========================================================================
    # 生命周期
    # =========================================================================

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None


class MCPError(Exception):
    """MCP 调用错误"""
    pass


# =============================================================================
# 全局实例
# =============================================================================

_wiki_mcp: Optional[WikiMCPClient] = None


def get_wiki_mcp() -> WikiMCPClient:
    """获取Wiki MCP 客户端实例"""
    global _wiki_mcp
    if _wiki_mcp is None:
        from src.config import get_settings
        settings = get_settings()

        _wiki_mcp = WikiMCPClient(
            wiki_url=settings.wiki.mcp_read_url or "https://open.wiki.cn/mcp/stream/mcp_zohhcjEYgBomUF1c8lLwwQD7IOpvAzuSxRniVoF6y7mR0dBTq-12J50jKSLFrFNxu0aEYPaJ2uo",
            wiki_space_id=settings.wiki.wiki_space_id or "7065713946103119900",
        )
    return _wiki_mcp


def reset_wiki_mcp():
    """重置Wiki MCP 客户端"""
    global _wiki_mcp
    if _wiki_mcp:
        asyncio.create_task(_wiki_mcp.close())
    _wiki_mcp = None
