# -*- coding: utf-8 -*-
"""
统一LLM客户端

支持代理模式和直连模式，使用OpenAI兼容SDK。
"""

import json
import logging
import asyncio
from typing import Any, Dict, List, Optional, Union

from openai import AsyncOpenAI

from src.config import get_settings, LLMConfig

logger = logging.getLogger(__name__)


class UnifiedLLMClient:
    """统一LLM客户端"""

    def __init__(self, config: Optional[LLMConfig] = None):
        if config is None:
            config = get_settings().llm
        self.config = config
        self._client: Optional[AsyncOpenAI] = None
        self._init_client()

    def _init_client(self):
        base_url = self.config.get_effective_url()
        api_key = self.config.get_effective_key()

        if self.config.use_proxy:
            logger.info(f"LLM客户端: 代理模式 -> {base_url}")
        else:
            logger.info(f"LLM客户端: 直连模式 -> {base_url}")
            if not api_key:
                logger.warning("API Key未配置")

        if api_key:
            self._client = AsyncOpenAI(
                base_url=base_url,
                api_key=api_key,
                timeout=self.config.timeout,
                max_retries=0
            )
        else:
            self._client = None

    @property
    def is_configured(self) -> bool:
        return self._client is not None

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        return_usage: bool = False,
        **kwargs
    ) -> Union[str, Dict[str, Any]]:
        """
        发送聊天请求

        支持:
        - 文本消息: {"role": "user", "content": "hello"}
        - 多模态消息: {"role": "user", "content": [{"type": "text", "text": "..."}, {"type": "image_url", "image_url": {"url": "..."}}]}
        """
        if not self._client:
            raise RuntimeError("LLM未配置！请检查配置中的 API Key 和 base_url 设置。")

        try:
            request_params = {
                "model": model or self.config.model,
                "messages": messages,
                "temperature": temperature if temperature is not None else self.config.temperature,
                "max_tokens": max_tokens or self.config.max_tokens,
            }

            if tools:
                request_params["tools"] = tools
                if tool_choice:
                    request_params["tool_choice"] = tool_choice

            # 空响应重试（abort 通常是上下文过长或服务端问题）
            for _abort_try in range(3):
                response = await asyncio.wait_for(
                    self._client.chat.completions.create(**request_params),
                    timeout=self.config.timeout or 600,
                )
                _fr = response.choices[0].finish_reason if response.choices else None
                _msg = response.choices[0].message if response.choices else None
                _ct = getattr(response.usage, 'completion_tokens', None) if response.usage else None
                _has_tc = bool(getattr(_msg, 'tool_calls', None)) if _msg else False
                _has_content = bool(getattr(_msg, 'content', None)) if _msg else False
                if (not _ct or _ct == 0) and not _has_tc and not _has_content:
                    delay = [3, 5, 8][_abort_try]
                    logger.warning(f"LLM空响应(finish_reason={_fr})，第{_abort_try+1}/3次重试，等{delay}s...")
                    await asyncio.sleep(delay)
                    continue
                break

            usage = None
            if hasattr(response, 'usage') and response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }

            message = response.choices[0].message

            if message.tool_calls:
                response_dict = response.model_dump()
                tool_calls_dict = response_dict["choices"][0]["message"].get("tool_calls", [])

                function_calls = []
                mcp_calls = []

                for tc_dict in tool_calls_dict:
                    tc_type = tc_dict.get("type")
                    if tc_type == "mcp":
                        mcp_calls.append(tc_dict)
                    elif tc_type == "function" and "function" in tc_dict:
                        function_calls.append(tc_dict)

                if function_calls:
                    tool_names = [fc["function"].get("name", "") for fc in function_calls]
                    logger.info(f"LLM请求调用工具: {tool_names}")
                    result = json.dumps({
                        "type": "tool_calls",
                        "tool_calls": function_calls
                    }, ensure_ascii=False)
                    if return_usage:
                        return {"content": result, "usage": usage}
                    return result

                if mcp_calls:
                    mcp_outputs = []
                    for tc_dict in mcp_calls:
                        mcp_data = tc_dict.get("mcp", {})
                        if mcp_data.get("type") == "mcp_call":
                            mcp_output = mcp_data.get("output", "")
                            if mcp_output:
                                mcp_outputs.append(str(mcp_output))

                    result_parts = []
                    if message.content:
                        result_parts.append(message.content)
                    if mcp_outputs:
                        result_parts.append("\n\n--- MCP 工具原始数据 ---\n" + "\n".join(mcp_outputs))

                    combined = "\n".join(result_parts) if result_parts else ""
                    if return_usage:
                        return {"content": combined, "usage": usage}
                    return combined

            content = message.content or ""

            if return_usage:
                return {"content": content, "usage": usage}
            return content

        except Exception as e:
            error_msg = str(e)

            # 诊断日志：500 时打印请求体关键信息
            if "500" in error_msg:
                n_msgs = len(messages)
                total_chars = sum(len(str(m.get("content", ""))) for m in messages)
                n_tools = len(tools) if tools else 0
                # 检查 messages 中是否有异常结构
                last_msg = messages[-1] if messages else {}
                last_role = last_msg.get("role", "?")
                last_tc = bool(last_msg.get("tool_calls"))
                # 检查是否有连续相同 role
                roles = [m.get("role") for m in messages[-6:]]
                logger.warning(
                    f"[500诊断] msgs={n_msgs}, chars={total_chars}, tools={n_tools}, "
                    f"last_role={last_role}, last_has_tc={last_tc}, recent_roles={roles}"
                )
                # 检查 tool_calls 中是否有 None id
                for i, m in enumerate(messages):
                    if m.get("role") == "assistant" and m.get("tool_calls"):
                        for tc in m["tool_calls"]:
                            if not tc.get("id"):
                                logger.error(f"[500诊断] msg[{i}] tool_call 缺少 id: {tc}")
                    if m.get("role") == "tool" and not m.get("tool_call_id"):
                        logger.error(f"[500诊断] msg[{i}] tool response 缺少 tool_call_id")

            # 429 Rate Limit - 短暂退避重试
            if "429" in error_msg or "速率限制" in error_msg:
                for retry_idx in range(3):
                    delay = [3, 6, 12][retry_idx]
                    logger.warning(f"LLM请求限流(429)，第{retry_idx+1}/3次重试，等{delay}秒...")
                    await asyncio.sleep(delay)
                    try:
                        response = await self._client.chat.completions.create(**request_params)
                        message = response.choices[0].message
                        usage = None
                        if hasattr(response, 'usage') and response.usage:
                            usage = {"prompt_tokens": response.usage.prompt_tokens, "completion_tokens": response.usage.completion_tokens, "total_tokens": response.usage.total_tokens}
                        if message.tool_calls:
                            response_dict = response.model_dump()
                            tool_calls_dict = response_dict["choices"][0]["message"].get("tool_calls", [])
                            function_calls = [tc for tc in tool_calls_dict if tc and tc.get("type") == "function"]
                            if function_calls:
                                result = json.dumps({"type": "tool_calls", "tool_calls": function_calls}, ensure_ascii=False)
                                logger.info(f"LLM限流重试成功(第{retry_idx+1}次)")
                                return {"content": result, "usage": usage} if return_usage else result
                        content = message.content or ""
                        logger.info(f"LLM限流重试成功(第{retry_idx+1}次)")
                        return {"content": content, "usage": usage} if return_usage else content
                    except Exception as retry_e:
                        if "429" not in str(retry_e) and "速率" not in str(retry_e):
                            logger.error(f"LLM限流重试失败(非429): {retry_e}")
                            raise retry_e

            # 400/1210 错误 - 重试1次
            if "400" in error_msg and "1210" in error_msg:
                logger.warning(f"LLM请求失败(400/1210)，等5秒重试... 原因: {error_msg[:150]}")
                await asyncio.sleep(5)
                try:
                    response = await self._client.chat.completions.create(**request_params)
                    message = response.choices[0].message
                    usage = None
                    if hasattr(response, 'usage') and response.usage:
                        usage = {"prompt_tokens": response.usage.prompt_tokens, "completion_tokens": response.usage.completion_tokens, "total_tokens": response.usage.total_tokens}
                    if message.tool_calls:
                        response_dict = response.model_dump()
                        tool_calls_dict = response_dict["choices"][0]["message"].get("tool_calls", [])
                        function_calls = [tc for tc in tool_calls_dict if tc and tc.get("type") == "function"]
                        if function_calls:
                            result = json.dumps({"type": "tool_calls", "tool_calls": function_calls}, ensure_ascii=False)
                            return {"content": result, "usage": usage} if return_usage else result
                    content = message.content or ""
                    return {"content": content, "usage": usage} if return_usage else content
                except Exception as retry_e:
                    logger.error(f"LLM 400重试也失败: {retry_e}")

            # 500错误重试
            if "500" in error_msg:
                max_retries = 3
                retry_delays = [2, 4, 8]
                for retry_idx in range(max_retries):
                    delay = retry_delays[retry_idx] if retry_idx < len(retry_delays) else 15
                    logger.warning(f"LLM请求失败(500)，第{retry_idx+1}/{max_retries}次重试，等{delay}秒...")
                    await asyncio.sleep(delay)
                    try:
                        retry_params = {
                            "model": model or self.config.model,
                            "messages": messages,
                            "temperature": temperature if temperature is not None else self.config.temperature,
                            "max_tokens": max_tokens or self.config.max_tokens,
                        }
                        if tools and retry_idx < max_retries - 1:
                            retry_params["tools"] = tools
                            if tool_choice:
                                retry_params["tool_choice"] = tool_choice
                        response = await self._client.chat.completions.create(**retry_params)
                        message = response.choices[0].message
                        if message.tool_calls:
                            response_dict = response.model_dump()
                            tool_calls_dict = response_dict["choices"][0]["message"].get("tool_calls", [])
                            function_calls = [tc for tc in tool_calls_dict if tc and tc.get("type") == "function"]
                            if function_calls:
                                result = json.dumps({"type": "tool_calls", "tool_calls": function_calls}, ensure_ascii=False)
                                if return_usage:
                                    u = None
                                    if hasattr(response, 'usage') and response.usage:
                                        u = {"prompt_tokens": response.usage.prompt_tokens, "completion_tokens": response.usage.completion_tokens, "total_tokens": response.usage.total_tokens}
                                    return {"content": result, "usage": u}
                                return result
                        content = message.content or ""
                        if return_usage:
                            u = None
                            if hasattr(response, 'usage') and response.usage:
                                u = {"prompt_tokens": response.usage.prompt_tokens, "completion_tokens": response.usage.completion_tokens, "total_tokens": response.usage.total_tokens}
                            return {"content": content, "usage": u}
                        return content
                    except Exception as retry_e:
                        logger.warning(f"LLM第{retry_idx+1}次重试失败: {retry_e}")

            err_lower = str(e).lower()
            if any(kw in err_lower for kw in ["timeout", "timed out", "connection", "reset"]):
                logger.warning(f"LLM请求超时/网络错误: {e}")
            else:
                # 不打 traceback，避免日志膨胀（错误信息本身已足够诊断）
                logger.error(f"LLM请求失败: {e}")
            raise

    @staticmethod
    def mcp_tool_to_openai_function(mcp_tool: Dict[str, Any]) -> Dict[str, Any]:
        """将MCP工具格式转换为OpenAI Function格式"""
        return {
            "type": "function",
            "function": {
                "name": mcp_tool.get("name", ""),
                "description": mcp_tool.get("description", ""),
                "parameters": mcp_tool.get("inputSchema", {
                    "type": "object",
                    "properties": {},
                    "required": []
                })
            }
        }


# 全局客户端实例
_llm_client: Optional[UnifiedLLMClient] = None


def get_llm_client() -> UnifiedLLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = UnifiedLLMClient()
    return _llm_client


def reset_llm_client():
    global _llm_client
    _llm_client = None


# ---------------------------------------------------------------------------
# Anthropic 适配器 — 兼容 UnifiedLLMClient.chat() 返回格式
# ---------------------------------------------------------------------------

class AnthropicLLMClient:
    """Anthropic Messages API 适配器，返回格式与 UnifiedLLMClient.chat() 一致。

    用于 eval 框架中替换 GLM 客户端，让 agent_runner 无需修改。
    """

    def __init__(self, base_url: str, api_key: str, model: str, timeout: int = 120):
        import httpx
        self.model = model
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=timeout,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
        )

    def _convert_tools_to_anthropic(self, tools: List[Dict]) -> List[Dict]:
        """OpenAI function schema → Anthropic tool schema"""
        result = []
        for t in tools:
            func = t.get("function", t)
            result.append({
                "name": func.get("name", ""),
                "description": func.get("description", ""),
                "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
            })
        return result

    def _convert_messages_to_anthropic(self, messages: List[Dict]) -> tuple:
        """OpenAI messages → (system, anthropic_messages)

        处理 system 消息提取、tool_calls/tool 消息转换。
        """
        system = ""
        anthropic_msgs = []

        for msg in messages:
            role = msg.get("role", "")

            if role == "system":
                system = msg.get("content", "")
                continue

            if role == "user":
                anthropic_msgs.append({"role": "user", "content": msg.get("content", "")})
                continue

            if role == "assistant":
                content_blocks = []
                text = msg.get("content", "")
                if text:
                    # text 可能是 JSON 字符串（tool_calls 编码），也可能是纯文本
                    try:
                        parsed = json.loads(text) if isinstance(text, str) else text
                        if isinstance(parsed, dict) and parsed.get("type") == "tool_calls":
                            # 从编码的 tool_calls JSON 还原
                            for tc in parsed.get("tool_calls", []):
                                func = tc.get("function", {})
                                args_str = func.get("arguments", "{}")
                                try:
                                    args = json.loads(args_str) if isinstance(args_str, str) else args_str
                                except json.JSONDecodeError:
                                    args = {}
                                content_blocks.append({
                                    "type": "tool_use",
                                    "id": tc.get("id", f"toolu_{id(tc)}"),
                                    "name": func.get("name", ""),
                                    "input": args,
                                })
                        else:
                            content_blocks.append({"type": "text", "text": text})
                    except (json.JSONDecodeError, TypeError):
                        content_blocks.append({"type": "text", "text": text})

                # 也处理直接的 tool_calls 列表（agent_runner 格式）
                if msg.get("tool_calls"):
                    for tc in msg["tool_calls"]:
                        func = tc.get("function", {})
                        args_str = func.get("arguments", "{}")
                        try:
                            args = json.loads(args_str) if isinstance(args_str, str) else args_str
                        except json.JSONDecodeError:
                            args = {}
                        content_blocks.append({
                            "type": "tool_use",
                            "id": tc.get("id", f"toolu_{id(tc)}"),
                            "name": func.get("name", ""),
                            "input": args,
                        })

                if content_blocks:
                    anthropic_msgs.append({"role": "assistant", "content": content_blocks})
                continue

            if role == "tool":
                # tool response → user message with tool_result
                # 合并连续的 tool results 到同一个 user 消息
                tool_result_block = {
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id", ""),
                    "content": msg.get("content", ""),
                }
                if anthropic_msgs and anthropic_msgs[-1]["role"] == "user":
                    last_content = anthropic_msgs[-1]["content"]
                    if isinstance(last_content, list) and last_content and last_content[0].get("type") == "tool_result":
                        # 追加到已有的 tool_result user 消息
                        last_content.append(tool_result_block)
                        continue
                anthropic_msgs.append({
                    "role": "user",
                    "content": [tool_result_block],
                })
                continue

        return system, anthropic_msgs

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        return_usage: bool = False,
        **kwargs
    ) -> Union[str, Dict[str, Any]]:
        """兼容 UnifiedLLMClient.chat() 的接口"""

        system, anthropic_msgs = self._convert_messages_to_anthropic(messages)

        payload: Dict[str, Any] = {
            "model": model or self.model,
            "max_tokens": max_tokens or 8192,
            "messages": anthropic_msgs,
        }
        if system:
            payload["system"] = system
        if temperature is not None:
            payload["temperature"] = temperature
        if tools:
            payload["tools"] = self._convert_tools_to_anthropic(tools)

        # 重试逻辑
        for attempt in range(6):
            try:
                resp = await self._client.post("/v1/messages", json=payload)
                if resp.status_code in (429, 529):
                    delay = min(5 * (2 ** attempt), 60)
                    logger.warning(f"Anthropic agent {resp.status_code}，第{attempt+1}次，等{delay}s...")
                    await asyncio.sleep(delay)
                    continue
                resp.raise_for_status()
                break
            except Exception as e:
                if "429" in str(e) or "529" in str(e):
                    delay = min(5 * (2 ** attempt), 60)
                    await asyncio.sleep(delay)
                    continue
                raise
        else:
            raise RuntimeError("Anthropic API 限流重试耗尽")

        data = resp.json()

        # 解析 usage
        usage = None
        if data.get("usage"):
            u = data["usage"]
            usage = {
                "prompt_tokens": u.get("input_tokens", 0),
                "completion_tokens": u.get("output_tokens", 0),
                "total_tokens": u.get("input_tokens", 0) + u.get("output_tokens", 0),
            }

        # 解析 content blocks
        content_blocks = data.get("content", [])
        tool_use_blocks = [b for b in content_blocks if b.get("type") == "tool_use"]
        text_blocks = [b for b in content_blocks if b.get("type") == "text"]

        if tool_use_blocks:
            # 转换为 OpenAI 格式的 tool_calls
            function_calls = []
            for b in tool_use_blocks:
                function_calls.append({
                    "id": b.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": b.get("name", ""),
                        "arguments": json.dumps(b.get("input", {}), ensure_ascii=False),
                    },
                })
            tool_names = [fc["function"]["name"] for fc in function_calls]
            logger.info(f"LLM请求调用工具: {tool_names}")
            result = json.dumps({"type": "tool_calls", "tool_calls": function_calls}, ensure_ascii=False)
            if return_usage:
                return {"content": result, "usage": usage}
            return result

        # 纯文本
        text = "\n".join(b.get("text", "") for b in text_blocks)
        if return_usage:
            return {"content": text, "usage": usage}
        return text
