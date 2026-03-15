# -*- coding: utf-8 -*-
"""
Wiki API Relay & File Export Service
支持用户通过 /relay/login 自动完成 OAuth 授权，并由服务器代为执行文件导出。
"""

import asyncio
import json
import logging
import os
import time
import secrets
import urllib.parse
from pathlib import Path
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Request, Query
from fastapi.responses import RedirectResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
import httpx
import yaml

logger = logging.getLogger(__name__)

router = APIRouter()

# ── 配置 ──

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RELAY_TOKEN_FILE = DATA_DIR / "relay_tokens.json"
WIKI_API_BASE = "https://open.wiki.cn/open-apis"

# ── 模型 ──

class UserToken(BaseModel):
    user_access_token: str
    refresh_token: Optional[str] = None
    expires_at: float
    username: str
    ip: str
    last_updated: float

class RegisterRequest(BaseModel):
    user_access_token: str
    refresh_token: Optional[str] = None
    expires_in: int = 7200
    username: str = "unknown"

# ── 持久化管理器 ──

class RelayTokenManager:
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.tokens: Dict[str, UserToken] = {}
        self._load()

    def _load(self):
        if self.file_path.exists():
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.tokens = {ip: UserToken(**v) for ip, v in data.items()}
            except Exception as e:
                logger.error(f"Failed to load relay tokens: {e}")

    def _save(self):
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump({ip: v.model_dump() for ip, v in self.tokens.items()}, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save relay tokens: {e}")

    def get_token(self, ip: str) -> Optional[UserToken]:
        token = self.tokens.get(ip)
        if token and time.time() < token.expires_at - 300:
            return token
        return None

    def register(self, ip: str, access_token: str, refresh_token: str, expires_in: int, username: str):
        self.tokens[ip] = UserToken(
            user_access_token=access_token,
            refresh_token=refresh_token,
            expires_at=time.time() + expires_in,
            username=username,
            ip=ip,
            last_updated=time.time()
        )
        self._save()

    def unregister(self, ip: str):
        if ip in self.tokens:
            del self.tokens[ip]
            self._save()
            return True
        return False

token_manager = RelayTokenManager(RELAY_TOKEN_FILE)

# ── 辅助函数 ──

async def _get_wiki_config():
    """获取主应用的 app_id / app_secret"""
    config_path = Path(__file__).resolve().parents[2] / "config" / "default.yaml"
    if not config_path.exists():
        return None, None
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    wiki = cfg.get("wiki", {})
    return wiki.get("app_id"), wiki.get("app_secret")

async def _refresh_user_token_if_needed(ip: str) -> Optional[str]:
    """刷新 Token"""
    user_info = token_manager.tokens.get(ip)
    if not user_info:
        return None
    
    # 提前 30 分钟 (1800秒) 刷新，保证 token 随时处于高可用状态
    if time.time() < user_info.expires_at - 1800:
        return user_info.user_access_token
    
    if not user_info.refresh_token:
        logger.warning(f">>> [Relay] Cannot refresh token for {ip}, no refresh_token found.")
        return None

    app_id, app_secret = await _get_wiki_config()
    if not app_id or not app_secret:
        return None

    try:
        logger.info(f">>> [Relay] Auto-refreshing access token for IP: {ip} (User: {user_info.username})")
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{WIKI_API_BASE}/authen/v2/oauth/token",
                headers={"Content-Type": "application/json"},
                json={
                    "grant_type": "refresh_token",
                    "refresh_token": user_info.refresh_token,
                    "client_id": app_id,
                    "client_secret": app_secret,
                }
            )
            data = resp.json()
            if resp.status_code == 200 and data.get("code") == 0:
                token_data = data.get("data", {})
                token_manager.register(
                    ip=ip,
                    access_token=token_data.get("access_token"),
                    refresh_token=token_data.get("refresh_token", user_info.refresh_token),
                    expires_in=token_data.get("expires_in", 7200),
                    username=user_info.username
                )
                logger.info(f">>> [Relay] Successfully refreshed token for {ip}")
                return token_data.get("access_token")
            else:
                logger.error(f">>> [Relay] Failed to refresh token for {ip}: {data}")
    except Exception as e:
        logger.error(f">>> [Relay] Exception during token refresh for {ip}: {e}")
    
    return None

# ── 后台保活任务 ──

async def token_refresh_daemon():
    """每 5 分钟后台遍历一次所有 token，执行自动续期"""
    logger.info(">>> [Relay] Starting background token refresh daemon...")
    while True:
        try:
            await asyncio.sleep(300)  # 5分钟
            for ip in list(token_manager.tokens.keys()):
                await _refresh_user_token_if_needed(ip)
        except Exception as e:
            logger.error(f">>> [Relay] Error in token refresh daemon: {e}")

@router.on_event("startup")
async def start_background_tasks():
    asyncio.create_task(token_refresh_daemon())

# ── 核心授权路由 ──

@router.get("/login")
async def relay_login(request: Request):
    """[入口] 引导用户跳转到Wiki授权页"""
    logger.info(f">>> [Relay] Entering relay_login from IP: {request.client.host}")
    app_id, _ = await _get_wiki_config()
    if not app_id:
        logger.error(">>> [Relay] App ID not configured!")
        raise HTTPException(status_code=500, detail="App ID not configured")
    
    # 硬编码 redirect_uri 确保与Wiki后台配置完全一致
    redirect_uri = "http://10.3.39.132:8000/relay/callback"
    
    logger.info(f">>> [Relay] Using FIXED redirect_uri: {redirect_uri}")
    
    state = secrets.token_urlsafe(16)
    scopes = "wiki:wiki offline_access drive:export:readonly docs:document:export sheets:spreadsheet:readonly"
    
    params = {
        "client_id": app_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": scopes,
        "state": state
    }
    auth_url = f"https://accounts.wiki.cn/open-apis/authen/v1/authorize?{urllib.parse.urlencode(params)}"
    
    logger.info(f">>> [Relay] Redirecting to Wiki: {auth_url}")
    
    # 强制不缓存，使用 303 状态码确保浏览器执行跳转
    response = RedirectResponse(url=auth_url, status_code=303)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

@router.get("/callback")
async def relay_callback(request: Request, code: str, state: Optional[str] = None):
    """[回调] Wiki授权后跳回这里，换取 token 并绑定 IP"""
    ip = request.client.host
    app_id, app_secret = await _get_wiki_config()
    
    # 必须与 login 时完全一致
    redirect_uri = "http://10.3.39.132:8000/relay/callback"
    
    logger.info(f">>> [Relay] Callback from IP: {ip}, code: {code[:5]}..., using redirect_uri: {redirect_uri}")

    async with httpx.AsyncClient(timeout=30) as client:
        # 1. 换取 User Token
        token_payload = {
            "grant_type": "authorization_code",
            "code": code,
            "client_id": app_id,
            "client_secret": app_secret,
            "redirect_uri": redirect_uri,
        }
        token_resp = await client.post(
            f"{WIKI_API_BASE}/authen/v2/oauth/token",
            headers={"Content-Type": "application/json"},
            json=token_payload
        )
        data = token_resp.json()
        
        # Wiki authen/v2/oauth/token 接口返回的 token 是直接在根节点的
        # 或者在某些版本中在 data 节点，这里做兼容处理
        token_data = data.get("data") if "data" in data else data
        access_token = token_data.get("access_token")
        
        if not access_token:
            logger.error(f">>> [Relay] Token exchange failed! Response: {data}")
            return JSONResponse(status_code=400, content={"error": "No access_token in response", "details": data})
        
        logger.info(">>> [Relay] Token exchange successful")
        
        # 2. 获取用户信息以拿到用户名
        user_info_resp = await client.get(
            f"{WIKI_API_BASE}/authen/v1/user_info",
            headers={"Authorization": f"Bearer {access_token}"}
        )
        user_data = user_info_resp.json().get("data", {})
        username = user_data.get("name") or user_data.get("en_name") or "WikiUser"

        # 3. 注册并保存
        token_manager.register(
            ip=ip,
            access_token=access_token,
            refresh_token=token_data.get("refresh_token"),
            expires_in=token_data.get("expires_in", 7200),
            username=username
        )
        
        logger.info(f">>> [Relay] Successfully registered user: {username} for IP: {ip}")

        return {
            "message": "Authentication successful",
            "username": username,
            "ip": ip,
            "status": "Ready to use /relay/download"
        }

# ── 其它路由 ──

@router.post("/register")
async def register_relay(request: Request, req: RegisterRequest):
    """手动注册 Token (备选方案)"""
    ip = request.client.host
    token_manager.register(
        ip=ip,
        access_token=req.user_access_token,
        refresh_token=req.refresh_token,
        expires_in=req.expires_in,
        username=req.username
    )
    return {"message": f"Successfully registered tokens for {ip}", "username": req.username}

@router.post("/unregister")
async def unregister_relay(request: Request):
    ip = request.client.host
    if token_manager.unregister(ip):
        return {"message": f"Successfully unregistered {ip}"}
    raise HTTPException(status_code=404, detail="No registration found for this IP")

@router.get("/status")
async def get_relay_status(request: Request):
    ip = request.client.host
    
    # 主动尝试刷新，如果快过期的话（内部阈值已设为30分钟）
    await _refresh_user_token_if_needed(ip)
    
    # 重新获取刷新后的状态
    user_info = token_manager.tokens.get(ip)
    if not user_info:
        return {"registered": False, "ip": ip}
    
    remaining = user_info.expires_at - time.time()
    return {
        "registered": True,
        "ip": ip,
        "username": user_info.username,
        "expires_in": int(remaining),
        "is_valid": remaining > 300,
        "has_refresh_token": bool(user_info.refresh_token)
    }

@router.get("/download")
async def relay_download_file(
    request: Request,
    url: str = Query(..., description="Wiki文档或文件的 URL"),
    type: Optional[str] = Query(None, description="手动指定导出类型: docx/pdf/xlsx")
):
    """
    代用户下载文件。
    1. 自动检查 Token，若无则跳转 /relay/login
    2. 智能解析各种Wiki URL (Wiki/Docx/Sheet/Bitable)
    3. 自动匹配导出格式
    """
    ip = request.client.host
    token = await _refresh_user_token_if_needed(ip)
    
    # --- 体验优化: 如果没登录，直接跳到登录页 ---
    if not token:
        logger.info(f">>> [Relay] No token for {ip}, redirecting to login...")
        login_url = f"{request.url.scheme}://{request.headers.get('host', '127.0.0.1:8000')}/relay/login"
        return RedirectResponse(url=login_url)

    obj_token = ""
    obj_type = ""
    
    # --- 1. 智能解析 URL ---
    clean_url = url.split("?")[0].rstrip("/")
    
    if "/wiki/" in clean_url:
        wiki_token = clean_url.split("/wiki/")[-1].split("/")[-1]
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                f"{WIKI_API_BASE}/wiki/v2/spaces/get_node",
                headers={"Authorization": f"Bearer {token}"},
                params={"token": wiki_token}
            )
            node_data = resp.json()
            if node_data.get("code") != 0:
                raise HTTPException(status_code=400, detail=f"Wiki Wiki Error: {node_data.get('msg')}")
            node = node_data.get("data", {}).get("node", {})
            obj_token = node.get("obj_token")
            obj_type = node.get("obj_type")
    elif "/docx/" in clean_url:
        obj_token = clean_url.split("/docx/")[-1].split("/")[-1]
        obj_type = "docx"
    elif "/sheets/" in clean_url:
        obj_token = clean_url.split("/sheets/")[-1].split("/")[-1]
        obj_type = "sheet"
    elif "/base/" in clean_url:
        # 多维表格
        obj_token = clean_url.split("/base/")[-1].split("/")[-1]
        obj_type = "bitable"
    elif "/docs/" in clean_url:
        obj_token = clean_url.split("/docs/")[-1].split("/")[-1]
        obj_type = "doc"
    
    if not obj_token:
        if len(url) > 20 and "/" not in url:
            obj_token = url
            obj_type = "docx"
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported URL format: {url}")

    # --- 2. 自动匹配导出格式 ---
    # 注意：在Wiki导出接口中，'type' 必须是 [doc, sheet, bitable, docx] 之一，指的是原对象的类型
    # 而 'file_extension' 才是目标格式 (docx/xlsx/pdf)
    
    export_ext = type
    if not export_ext:
        if obj_type in ("sheet", "bitable"):
            export_ext = "xlsx"
        else:
            export_ext = "docx"
    
    logger.info(f">>> [Relay] Start: obj_type={obj_type} -> target_ext={export_ext} (token: {obj_token[:5]}...)")

    headers = {"Authorization": f"Bearer {token}"}
    async with httpx.AsyncClient(timeout=120) as client:
        # 3. 创建导出任务
        export_resp = await client.post(
            f"{WIKI_API_BASE}/drive/v1/export_tasks",
            headers=headers,
            json={
                "token": obj_token, 
                "type": obj_type,  # 这里必须传对象真实的类型 (bitable/docx/sheet)
                "file_extension": export_ext # 这里传目标扩展名 (xlsx/docx)
            }
        )
        export_data = export_resp.json()
        if export_data.get("code") != 0:
            logger.error(f">>> [Relay] Export task creation failed: {export_data}")
            return JSONResponse(status_code=400, content={"error": "Wiki Export Task Failed", "details": export_data})

        ticket = export_data.get("data", {}).get("ticket")
        
        # 4. 轮询
        file_token = ""
        for _ in range(30):
            await asyncio.sleep(2)
            status_resp = await client.get(
                f"{WIKI_API_BASE}/drive/v1/export_tasks/{ticket}?token={obj_token}",
                headers=headers
            )
            res_data = status_resp.json()
            res = res_data.get("data", {}).get("result", {})
            if res.get("job_status") == 0:
                file_token = res.get("file_token")
                break
            elif res.get("job_status") not in (1, 2):
                logger.error(f">>> [Relay] Export job failed in polling: {res_data}")
                raise HTTPException(status_code=500, detail=f"Export job failed: {res_data.get('msg')}")

        if not file_token:
            raise HTTPException(status_code=504, detail="Timeout")

        # 5. 下载并转发
        download_url = f"{WIKI_API_BASE}/drive/v1/export_tasks/file/{file_token}/download"
        
        # 为了解决 StreamClosed 报错，我们需要手动控制流的生命周期
        # 更好的做法是下载到内存或由 StreamingResponse 自动关闭，但这里我们使用 httpx 的流式处理
        async def stream_generator():
            async with httpx.AsyncClient(timeout=300) as stream_client:
                async with stream_client.stream("GET", download_url, headers=headers, follow_redirects=True) as response:
                    if response.status_code != 200:
                        yield b"Download from Wiki failed"
                        return
                    async for chunk in response.aiter_bytes():
                        yield chunk

        return StreamingResponse(
            stream_generator(),
            media_type="application/octet-stream"
        )

def get_relay_router():
    return router
