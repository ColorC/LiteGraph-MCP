# -*- coding: utf-8 -*-
"""Windows 文件桥接守护。

职责：
- 监控 WINDOWS_FILE_BRIDGE_URL 的 /health
- 健康失败时按需自动拉起桥接服务
- 在主服务生命周期内持续自愈
"""

import asyncio
import logging
import os
import subprocess
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)


class BridgeSupervisor:
    """Windows 文件桥接守护器。"""

    def __init__(self):
        self.bridge_url = (os.environ.get("WINDOWS_FILE_BRIDGE_URL") or "").rstrip("/")
        self.autostart = os.environ.get("WINDOWS_FILE_BRIDGE_AUTOSTART", "1") not in ("0", "false", "False")
        self.interval_sec = float(os.environ.get("WINDOWS_FILE_BRIDGE_HEALTH_INTERVAL_SEC", "8"))
        self.timeout_sec = float(os.environ.get("WINDOWS_FILE_BRIDGE_HEALTH_TIMEOUT_SEC", "2"))
        self.start_cmd = (os.environ.get("WINDOWS_FILE_BRIDGE_START_CMD") or "").strip()

        # 未配置 URL 时，自动回落到默认本地桥接地址
        if not self.bridge_url and self.autostart:
            self.bridge_url = "http://localhost:8765"
            os.environ["WINDOWS_FILE_BRIDGE_URL"] = self.bridge_url
            logger.info("[bridge_supervisor] WINDOWS_FILE_BRIDGE_URL 未配置，使用默认: %s", self.bridge_url)

        if not self.start_cmd and self.autostart:
            inferred = self._infer_default_start_cmd()
            if inferred:
                self.start_cmd = inferred

        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        self._last_health_ok: Optional[bool] = None

    @staticmethod
    def _infer_default_start_cmd() -> str:
        """推断默认桥接启动命令（仅使用本仓库 AgentClient 桥接脚本）。"""
        project_root = Path(__file__).resolve().parents[2]  # open_graph-graph-rag
        local_script = project_root / "scripts" / "start_file_bridge.sh"
        if local_script.exists():
            return f'bash "{local_script}"'
        return ""

    def _health_url(self) -> str:
        return f"{self.bridge_url}/health"

    def _check_health_sync(self) -> bool:
        if not self.bridge_url:
            return False
        try:
            health_resp = requests.get(self._health_url(), timeout=self.timeout_sec)
            if health_resp.status_code != 200:
                return False

            # 开发阶段强约束：必须具备 /read_file 协议，不允许仅 health 存活
            probe_resp = requests.post(
                f"{self.bridge_url}/read_file",
                json={"path": "__bridge_probe__.txt"},
                timeout=self.timeout_sec,
            )
            return probe_resp.status_code == 200
        except requests.RequestException:
            return False

    def _start_bridge_sync(self) -> bool:
        if not self.start_cmd:
            logger.warning("[bridge_supervisor] 未配置桥接启动命令，无法自动拉起")
            return False

        try:
            logger.warning("[bridge_supervisor] 检测到桥接不可用，尝试拉起: %s", self.start_cmd)
            result = subprocess.run(
                self.start_cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                stderr = (result.stderr or "").strip()
                logger.warning("[bridge_supervisor] 拉起命令失败(code=%s): %s", result.returncode, stderr)
                return False
            return True
        except Exception as e:
            logger.warning("[bridge_supervisor] 拉起桥接异常: %s", e)
            return False

    async def start(self):
        if not self.bridge_url:
            logger.info("[bridge_supervisor] 未配置 WINDOWS_FILE_BRIDGE_URL，桥接守护未启用")
            return

        if self._task and not self._task.done():
            return

        self._stop_event.clear()
        self._task = asyncio.create_task(self._run_loop(), name="bridge-supervisor")
        logger.info("[bridge_supervisor] 已启动，监控地址: %s", self._health_url())

    async def stop(self):
        if not self._task:
            return
        self._stop_event.set()
        await self._task
        self._task = None
        logger.info("[bridge_supervisor] 已停止")

    async def _run_loop(self):
        loop = asyncio.get_event_loop()

        while not self._stop_event.is_set():
            healthy = await loop.run_in_executor(None, self._check_health_sync)

            if self._last_health_ok is None or healthy != self._last_health_ok:
                state = "healthy" if healthy else "unhealthy"
                logger.info("[bridge_supervisor] health=%s", state)
                self._last_health_ok = healthy

            if not healthy and self.autostart:
                started = await loop.run_in_executor(None, self._start_bridge_sync)
                if started:
                    await asyncio.sleep(1)
                    healthy_after = await loop.run_in_executor(None, self._check_health_sync)
                    if healthy_after:
                        logger.info("[bridge_supervisor] 桥接服务拉起成功")
                    else:
                        logger.warning("[bridge_supervisor] 桥接服务拉起后仍不可用")

            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self.interval_sec)
            except asyncio.TimeoutError:
                pass
