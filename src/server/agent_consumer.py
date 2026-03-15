# -*- coding: utf-8 -*-
"""
ProposalConsumer — 后台消费已审批的 proposal，用 LLM agent 执行图变更。

工具集：
- 图谱: graph_search, graph_neighbors, graph_edit
- Wiki项目: search_stories, get_story_detail
- Wiki知识库: wiki_doc_reader
- 代码: search_code
"""

import asyncio
import concurrent.futures
import json
import logging
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Optional, List, Any

from src.tools.question_manager import QuestionManager
from src.ingest.tools import (
    GraphEditTool,
    GraphSearchTool,
    GraphNeighborsTool,
    SearchCodeTool,
    WikiSearchStoriesTool,
    WikiGetStoryDetailTool,
)
from src.ingest.agent_runner import run_agent_with_tools

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent  # extensions/open_graph-graph-rag

SYSTEM_PROMPT = """你是知识图谱维护 agent。你的任务是理解人类审批意见，对知识图谱执行恰当的变更。

## 可用工具

**图谱查询**
- graph_search: 搜索节点（hybrid/bm25/exact/label_list）
- graph_neighbors: 查看节点的邻居和边

**图谱编辑**
- graph_edit: 支持 update_node / add_node / remove_node / add_edge / remove_edge

**信息收集**
- search_code: 搜索代码文件
- search_stories: 搜索Wiki需求
- get_story_detail: 获取需求详情
- wiki_doc_reader: 读取Wiki文档

## 核心原则（优先级从高到低）

### 1. 证据驱动
- 修改前必须收集证据：用 search_code 查代码、search_stories 查需求、graph_neighbors 查关联
- 不要凭猜测操作。如果证据不足，先收集再行动
- 证据来源的优先级：代码 > Wiki需求 > 文档 > 图谱现有数据

### 2. 验证性
- 每次 graph_edit 后必须用 graph_search 或 graph_neighbors 验证结果
- 如果验证发现结果不符合预期，分析原因并修正

### 3. 彻底性
- 思考变更的完整影响范围：哪些节点、边、属性会受影响
- 不要只改表面。如果修改涉及概念归属的变化，检查并更新相关的边
- 但也不要过度修改——只做必要的变更

## 工作流程

1. **理解意图**: 仔细阅读人类意见，理解它要求的本质是什么
2. **收集证据**: 用搜索工具确认当前状态和正确状态
3. **规划变更**: 在脑中或输出中列出需要执行的操作序列
4. **执行变更**: 逐步执行，每步后验证
5. **总结输出**: 说明做了什么、为什么、影响了什么

## 常见场景的思考框架

**"X不应该是Y的别名/描述有误"**:
- 这个修改意味着什么概念关系的变化？
- 有哪些实体（entity）或边因为错误的别名而被错误关联？
- 用 search_code 确认代码中的实际命名和归属
- 修改属性后，检查并迁移错误关联的边

**"A和B应该是同一个/不应该关联"**:
- 收集两个节点的完整信息
- 确认边是否确实错误，还是只是看起来弱
- 删除或创建边后验证两端节点

**"信息缺失/需要补充"**:
- 先搜索确认缺失的信息确实不存在
- 补充后验证节点属性已更新

## 注意事项

- 节点 ID 规范: bt:术语名, entity:实体名_序号, wiki_story:ID
- 每次只做一个逻辑变更，验证后再继续
- 完成后输出变更摘要：做了什么、为什么、影响了哪些节点和边
"""


def _init_wiki_tools():
    """初始化Wiki项目 API 工具（通过子进程调用 open_graph_agent）"""
    try:
        wiki_agent_root = ROOT.parent.parent / "open_graph_agent"
        if not wiki_agent_root.exists():
            raise FileNotFoundError(f"未找到 open_graph_agent 目录: {wiki_agent_root}")
        python_exec = sys.executable

        def _call_tool(tool_name: str, payload: dict) -> dict:
            code = r"""
import json, sys
from pathlib import Path
agent_root = Path(sys.argv[1])
tool_name = sys.argv[2]
payload = json.loads(sys.argv[3])
sys.path.insert(0, str(agent_root))
from src.services.wiki_project.tools import search_by_mql, get_workitem_brief
if tool_name == "__health__":
    result = {"ok": True}
elif tool_name == "search_by_mql":
    result = search_by_mql(payload["moql"], **payload.get("kwargs", {}))
elif tool_name == "get_workitem_brief":
    result = get_workitem_brief(payload["work_item_id"], **payload.get("kwargs", {}))
else:
    result = {"error": f"unknown tool {tool_name}"}
print(json.dumps(result, ensure_ascii=False))
"""
            proc = subprocess.run(
                [python_exec, "-c", code, str(wiki_agent_root), tool_name,
                 json.dumps(payload, ensure_ascii=False)],
                capture_output=True, text=True, timeout=120,
            )
            if proc.returncode != 0:
                raise RuntimeError(proc.stderr.strip() or f"{tool_name} 执行失败")
            output = proc.stdout.strip()
            if not output:
                raise RuntimeError(f"{tool_name} 返回空结果")
            return json.loads(output)

        def search_fn(moql, **kwargs):
            return _call_tool("search_by_mql", {"moql": moql, "kwargs": kwargs})

        def detail_fn(work_item_id, **kwargs):
            return _call_tool("get_workitem_brief", {"work_item_id": work_item_id, "kwargs": kwargs})

        # 健康检查
        health = _call_tool("__health__", {})
        if not health.get("ok"):
            raise RuntimeError(f"Wiki工具健康检查失败: {health}")

        return WikiSearchStoriesTool(search_fn), WikiGetStoryDetailTool(detail_fn)
    except Exception as e:
        logger.warning(f"[Consumer] Wiki项目工具初始化失败（可选）: {e}")
        return None, None


class ProposalConsumer:
    """后台 agent：轮询 approved 状态的 proposal，用 LLM agent 执行图变更。"""

    def __init__(self, qm: QuestionManager):
        self.qm = qm
        self.running = False
        self.interval = 10
        self._processing = False
        self._retriever = None
        self._wiki_search = None
        self._wiki_detail = None
        self._tools_initialized = False
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="agent-consumer")

        self.stats = {
            "total_processed": 0,
            "completed": 0,
            "failed": 0,
            "no_action": 0,
        }

    def _init_tools(self):
        """延迟初始化工具（首次处理时调用）"""
        if self._tools_initialized:
            return

        # 初始化检索器（用于 GraphSearchTool 的语义搜索）
        try:
            from src.retrieval.hybrid_retriever import GraphRAGRetriever
            self._retriever = GraphRAGRetriever(self.qm.gm)
            logger.info("[Consumer] 混合检索器初始化成功")
        except Exception as e:
            logger.warning(f"[Consumer] 检索器初始化失败，graph_search 将降级: {e}")

        # 初始化Wiki工具
        self._wiki_search, self._wiki_detail = _init_wiki_tools()

        self._tools_initialized = True

    def _build_tools(self) -> List[Any]:
        """构建当前可用的工具列表"""
        self._init_tools()
        gm = self.qm.gm

        tools = [
            GraphEditTool(gm),
            GraphNeighborsTool(gm),
            SearchCodeTool(gm),
        ]

        # GraphSearchTool 需要 retriever
        if self._retriever:
            tools.append(GraphSearchTool(self._retriever, gm))
        else:
            # 降级：只用 gm 的基础搜索
            tools.append(GraphSearchTool(None, gm))

        # Wiki工具（可选）
        if self._wiki_search:
            tools.append(self._wiki_search)
        if self._wiki_detail:
            tools.append(self._wiki_detail)

        return tools

    def get_stats(self):
        return dict(self.stats)

    async def start(self):
        logger.info("ProposalConsumer started.")
        # 启动前清理“僵尸”任务（处于 in_progress 但实际已中断的任务）
        try:
            await self._cleanup_zombie_proposals()
        except Exception as e:
            logger.error(f"[Consumer] 清理僵尸任务失败: {e}")

        self.running = True
        while self.running:
            try:
                await self.process_pending_proposals()
            except Exception as e:
                logger.error(f"[Consumer] 轮询异常: {e}")
            await asyncio.sleep(self.interval)

    async def _cleanup_zombie_proposals(self):
        """将启动时处于 in_progress 状态的任务重置，防止死锁"""
        loop = asyncio.get_event_loop()
        all_questions = await loop.run_in_executor(
            None, self.qm.gm.find_nodes_by_label, "Question"
        )
        zombies = [q for q in all_questions if q.get("status") == "in_progress"]
        for q in zombies:
            pid = q["id"]
            logger.warning(f"[Consumer] 发现僵尸任务 {pid}，重置为 approved 待重试")
            self.qm.gm.update_node(pid, {
                "status": "approved",
                "execution_log": (q.get("execution_log") or "") + "\n[System] 检测到服务重启，任务已被自动重置。"
            })

    def stop(self):
        self.running = False
        logger.info("ProposalConsumer stopped.")

    async def process_pending_proposals(self):
        """处理所有 approved 状态的 proposal。

        所有同步操作（SQLite、subprocess、agent 执行）都在线程池中运行，
        避免阻塞 uvicorn event loop。
        """
        if self._processing:
            return
        self._processing = True

        try:
            # SQLite 查询放到线程池
            loop = asyncio.get_event_loop()
            all_questions = await loop.run_in_executor(
                None, self.qm.gm.find_nodes_by_label, "Question"
            )
            approved = [q for q in all_questions if q.get("status") == "approved"]

            for proposal in approved:
                pid = proposal["id"]
                self.stats["total_processed"] += 1

                try:
                    # 整个 proposal 处理在线程池中执行
                    result_info = await loop.run_in_executor(
                        self._executor,
                        self._process_single_proposal,
                        proposal,
                    )

                    if result_info["status"] == "no_action":
                        self.stats["no_action"] += 1
                    elif result_info["status"] == "completed":
                        self.stats["completed"] += 1
                        logger.info(f"[Consumer] 完成 {pid}: {result_info.get('summary', '')}")
                    elif result_info["status"] == "failed":
                        self.stats["failed"] += 1
                        logger.error(f"[Consumer] 执行失败 {pid}: {result_info.get('error', '')}")

                except Exception as e:
                    self.stats["failed"] += 1
                    logger.error(f"[Consumer] 执行失败 {pid}: {e}")
                    traceback.print_exc()

        finally:
            self._processing = False

    def _process_single_proposal(self, proposal: dict) -> dict:
        """在线程池中同步执行单个 proposal 的完整处理流程。

        包含 _init_tools（subprocess.run）、SQLite 读写、agent 执行，
        全部在工作线程中运行，不会阻塞主 event loop。
        """
        pid = proposal["id"]
        exec_log = []

        try:
            # 采纳且无意见：直接 pass，不走 LLM
            if proposal.get("auto_pass_no_review"):
                self.qm.gm.update_node(pid, {
                    "status": "completed_no_action",
                    "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "execution_result": "auto-pass(no refined answer)",
                    "execution_log": "审批无意见，按规则直接通过，未调用 LLM。",
                })
                return {"status": "no_action", "summary": "auto-pass(no refined answer)"}

            # 标记为处理中
            self.qm.gm.update_node(pid, {
                "status": "in_progress",
                "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            })

            # 构建 task
            task = self._build_task_prompt(proposal)
            if not task:
                self.qm.gm.update_node(pid, {
                    "status": "completed_no_action",
                    "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "execution_result": "无法构建任务（缺少关键信息），跳过",
                })
                return {"status": "no_action"}

            # 构建工具（含 _init_tools 的同步 subprocess 调用，现在安全地在线程中）
            tools = self._build_tools()
            tool_names = [t.name for t in tools]
            logger.info(f"[Consumer] 开始执行 {pid}, 工具: {tool_names}")

            def on_step(step, event_type, data):
                ts = time.strftime("%H:%M:%S")
                if event_type == "tool_call":
                    line = f"[{ts}] Step {step}: 调用 {data['tool']}({json.dumps(data['args'], ensure_ascii=False)[:200]})"
                elif event_type == "tool_result":
                    line = f"[{ts}] Step {step}: {data['tool']} → {data['result'][:300]}"
                elif event_type == "final":
                    line = f"[{ts}] Step {step}: 完成 — {data['content'][:300]}"
                else:
                    line = f"[{ts}] Step {step}: {event_type}"
                exec_log.append(line)
                # 实时写入节点（每 3 步或 final 时）
                if event_type == "final" or len(exec_log) % 3 == 0:
                    self.qm.gm.update_node(pid, {
                        "execution_log": "\n".join(exec_log[-50:]),
                    })

            # agent 需要 async event loop，在线程内创建独立的 loop
            thread_loop = asyncio.new_event_loop()
            try:
                # 增加 30 分钟强制超时，防止模型生成死循环或网络挂死
                async def run_with_timeout():
                    return await asyncio.wait_for(
                        run_agent_with_tools(
                            task=task,
                            tools=tools,
                            system_prompt=SYSTEM_PROMPT,
                            max_steps=40,
                            on_step=on_step,
                        ),
                        timeout=1800  # 30 minutes
                    )
                
                result = thread_loop.run_until_complete(run_with_timeout())
            finally:
                thread_loop.close()

            self.qm.gm.update_node(pid, {
                "status": "completed",
                "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "execution_result": result[:5000],
                "execution_log": "\n".join(exec_log),
            })
            return {"status": "completed", "summary": result[:200]}

        except Exception as e:
            self.qm.gm.update_node(pid, {
                "status": "failed",
                "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "execution_result": f"执行失败: {str(e)}",
                "execution_error": traceback.format_exc(),
            })
            return {"status": "failed", "error": str(e)}

    def _build_task_prompt(self, proposal: dict) -> Optional[str]:
        """根据 proposal 构建 agent 任务描述"""
        question = proposal.get("question", "")
        answer = proposal.get("answer", "")
        category = proposal.get("category", "")
        context = proposal.get("context", "")
        related_node_id = proposal.get("related_node_id", "")

        if not question:
            return None

        parts = [
            "## 待处理的图谱变更请求\n",
            f"**问题**: {question}\n",
        ]

        if category:
            parts.append(f"**分类**: {category}\n")

        if related_node_id:
            parts.append(f"**关联节点 ID**: {related_node_id}\n")

        if context:
            parts.append(f"**原始上下文**:\n{context[:3000]}\n")

        if answer:
            parts.append(f"\n## 人类审批意见（必须遵循）\n{answer}\n")
        else:
            parts.append("\n## 人类审批意见\n（人类同意了这个问题的建议，请按问题描述执行变更）\n")

        # 根据 category 给出思考方向（不是固定步骤）
        category_guidance = {
            "proposal": "人工提议的变更。重点：理解提议的本质意图，用代码/需求验证，思考完整影响范围。",
            "contradictory": "矛盾信息。重点：收集多方证据，判断哪个正确，再修改。",
            "wrong_association": "错误关联。重点：确认边确实错误后再删除，思考正确的关联目标。",
            "weak_association": "弱关联。重点：评估是否有保留价值，不要轻易删除可能有用的信息。",
        }
        guidance = category_guidance.get(category, "")
        if guidance:
            parts.append(f"\n## 思考方向（{category}）\n{guidance}\n")

        parts.append(
            "\n## 执行要求\n"
            "1. 理解人类意见的本质意图\n"
            "2. 收集证据（代码/需求/图谱）确认当前状态和目标状态\n"
            "3. 思考完整的影响范围，规划操作序列\n"
            "4. 逐步执行，每次操作后验证结果\n"
            "5. 输出变更摘要：做了什么、为什么、影响了哪些节点和边\n"
        )
        return "\n".join(parts)
