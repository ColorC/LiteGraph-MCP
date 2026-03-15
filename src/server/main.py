# -*- coding: utf-8 -*-
"""
OpenGraph Agent 后端代理服务

极简的API转发服务，用于隐藏API Key
支持OpenAI兼容格式的请求转发到GLM-4.7

使用方法:
    # 设置API Key
    set GLM_API_KEY=your_api_key_here
    
    # 启动服务
    python -m server.main
    
    # 或使用uvicorn
    uvicorn server.main:app --host 0.0.0.0 --port 8000
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.server.config import get_config
from src.server.routes import router
from src.server.relay import get_relay_router
# Assuming LightweightGraphManager is in src/graph/manager.py based on assumed structure
# I will verify this path in the next step if I guessed wrong, but standardizing on src.
from src.graph.manager import LightweightGraphManager 
from src.tools.question_manager import QuestionManager
from src.server.api import router as api_router, set_question_manager, set_proposal_consumer, set_edit_service
from src.server.bridge_supervisor import BridgeSupervisor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="OpenGraph Agent API Proxy",
    description="API Key保护的LLM请求转发服务 & Knowledge Graph API & Wiki Relay",
    version="1.2.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 添加CORS中间件（允许局域网访问）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
logger.info("正在注册核心路由...")
app.include_router(get_relay_router(), prefix="/relay")
app.include_router(router)
app.include_router(api_router, prefix="/api")

logger.info("路由注册完成: /relay, /, /api")



from src.server.agent_consumer import ProposalConsumer

# Global consumer instance
_consumer: Optional[ProposalConsumer] = None
_bridge_supervisor: Optional[BridgeSupervisor] = None


def init_app(gm: "LightweightGraphManager", qm: "QuestionManager"):
    """外部注入共享实例（由 server.py 统一入口调用）"""
    set_question_manager(qm)
    logger.info("QuestionManager 已注入（共享实例）")

    # 初始化编辑服务（与 stdio MCP 共享 graph_manager）
    import os
    from src.server.graph_edit_service import GraphEditService
    data_dir = os.environ.get("open_graph_DATA_DIR") or str(Path(__file__).resolve().parents[2] / "data")
    svc = GraphEditService(graph_manager=gm, data_dir=Path(data_dir))
    set_edit_service(svc)
    logger.info("GraphEditService 已注入（共享实例）")


@app.on_event("startup")
async def startup_event():
    """启动事件 — 如果已通过 init_app 注入则跳过独立初始化"""
    global _consumer, _bridge_supervisor
    config = get_config()

    logger.info("=" * 60)
    logger.info("OpenGraph Agent API Proxy 启动")
    logger.info("=" * 60)
    logger.info(f"监听地址: {config.host}:{config.port}")
    logger.info(f"GLM模型: {config.glm_model}")

    # 启动桥接守护（兼容直接运行 src.server.main）
    try:
        _bridge_supervisor = BridgeSupervisor()
        await _bridge_supervisor.start()
    except Exception:
        logger.exception("[bridge_supervisor] 在 FastAPI startup 中启动失败")

    # 如果已通过 init_app 注入，只启动 consumer
    from src.server.api import get_question_manager as _get_qm
    try:
        qm = _get_qm()
        # 已注入，启动 consumer
        _consumer = ProposalConsumer(qm)
        set_proposal_consumer(_consumer)
        import asyncio
        asyncio.create_task(_consumer.start())
        logger.info("ProposalConsumer 已启动（共享模式）")
        return
    except Exception:
        pass  # 未注入，走独立初始化

    # 独立启动模式（fallback: 直接 uvicorn src.server.main:app）
    graph_db_path = Path(config.graph_db_path)
    if not graph_db_path.exists():
        logger.warning(f"警告: 图数据库文件 {graph_db_path} 不存在，将尝试创建或等待初始化")
        graph_db_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"正在初始化图数据库: {graph_db_path}")
    try:
        from src.graph.manager import LightweightGraphManager
        gm = LightweightGraphManager(str(graph_db_path))
        qm = QuestionManager(gm)
        set_question_manager(qm)
        logger.info("QuestionManager 初始化成功（独立模式）")

        # Start Consumer
        _consumer = ProposalConsumer(qm)
        set_proposal_consumer(_consumer)
        import asyncio
        asyncio.create_task(_consumer.start())

    except Exception as e:
        logger.error(f"初始化失败: {e}")
        import traceback
        traceback.print_exc()

@app.on_event("shutdown")
async def shutdown_event():
    """关闭事件"""
    logger.info("OpenGraph Agent API Proxy 关闭")
    if _consumer:
        _consumer.stop()

    global _bridge_supervisor
    if _bridge_supervisor:
        try:
            await _bridge_supervisor.stop()
        except Exception:
            logger.exception("[bridge_supervisor] 在 FastAPI shutdown 中停止失败")



# 根路径 - API优先，静态文件兜底
# @app.get("/") -> Removed to allow StaticFiles to handle index.html

# Mount frontend — SPA catch-all for client-side routing
ui_dist_path = Path(__file__).resolve().parents[2] / "ui" / "dist"
if ui_dist_path.exists():
    from fastapi.responses import FileResponse

    # 静态资源（js/css/图片等）
    app.mount("/assets", StaticFiles(directory=str(ui_dist_path / "assets")), name="assets")

    # SPA catch-all: 所有非 /api 和 /relay 路径都返回 index.html
    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        # 更加健壮的排除逻辑
        path_segments = full_path.split("/")
        if path_segments[0] in ("api", "relay"):
            logger.info(f">>> [SPA] Skipping catch-all for potential API/Relay path: {full_path}")
            raise HTTPException(status_code=404)
            
        # 如果请求的是真实文件（如 favicon.ico），直接返回
        file_path = ui_dist_path / full_path
        if full_path and file_path.is_file():
            return FileResponse(str(file_path))
        return FileResponse(str(ui_dist_path / "index.html"))
else:
    logger.warning(f"UI build not found at {ui_dist_path}")
    @app.get("/")
    def root():
        return {"message": "API is running. UI building..."}

def main():
    """主函数"""
    import uvicorn
    
    config = get_config()
    
    uvicorn.run(
        "src.server.main:app", # Updated import path
        host=config.host,
        port=config.port,
        log_level=config.log_level.lower(),
        reload=False
    )

if __name__ == "__main__":
    main()
