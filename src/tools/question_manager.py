# -*- coding: utf-8 -*-
"""
问题管理器

Agent 在阅读文档/代码过程中产生的待人类确认问题的 CRUD 管理。
问题存储为图中的 Question 节点，通过 RAISES_QUESTION 边关联到产生问题的节点。

问题分类:
- ambiguous:     语义模糊，无代码可考证（可能是服务器逻辑/外部服务/未来规划）
- untraceable:   找不到相关资料，防止错误理解
- contradictory: 设计文档和实际代码不符

使用方式:
    qm = QuestionManager(graph_manager)
    qid = qm.create_question(
        question="BattleManager 文档中提到的'自动战斗'模式在代码中未发现，是否已删除？",
        category="contradictory",
        context="阅读 BattleManager 模块，对比 Script/Battle/BattleManager.cs",
        related_node_id="module:BattleManager"
    )
    pending = qm.list_pending(category="contradictory")
    qm.answer_question(qid, "自动战斗在 v2.3 移除了，现在用的是托管战斗")
"""

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Question 节点的标签
QUESTION_LABEL = "Question"

# Question → 源节点的边类型
RAISES_QUESTION_EDGE = "RAISES_QUESTION"

# 合法的问题分类（agent产生的 + UI proposal）
VALID_CATEGORIES = frozenset({"ambiguous", "untraceable", "contradictory", "game_trivia",
                              "weak_association", "wrong_association", "unknown",
                              "proposal", "general",
                              "ArchNode", "BusinessTerm", "CodeFile", "WikiStory",
                              "Folder", "InBusinessEntity", "Prefab", "Question"})

# 合法的状态
VALID_STATUSES = frozenset({"pending", "answered", "dismissed", "approved", "rejected"})


class QuestionManager:
    """
    问题管理器 —— 基于图数据库的 Question 节点 CRUD。

    不直接操作 SQLite，而是通过 LightweightGraphManager 的公共 API，
    这样 Question 节点自然参与到图的遍历和检索中。
    """

    def __init__(self, graph_manager):
        """
        Args:
            graph_manager: LightweightGraphManager 实例
        """
        self.gm = graph_manager
        self._counter = 0  # 用于生成唯一 ID

    # ========================================================================
    # 创建
    # ========================================================================

    def create_question(
        self,
        question: str,
        category: str,
        context: str,
        related_node_id: str,
        extra_props: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        创建一个 Question 节点并关联到产生问题的节点。

        Args:
            question:         问题内容
            category:         分类 (ambiguous / untraceable / contradictory / game_trivia)
            context:          问题产生的上下文（LLM 在阅读什么时产生的）
            related_node_id:  关联的节点 ID（产生问题的 BusinessTerm/Module/InBusinessEntity）
            extra_props:      额外属性
                - options: List[str] (for game_trivia)
                - correct_answer: str (for game_trivia)

        Returns:
            新创建的 Question 节点 ID
        """
        if category not in VALID_CATEGORIES:
            raise ValueError(
                f"无效的问题分类 '{category}', 必须是 {VALID_CATEGORIES} 之一"
            )

        # 生成唯一 ID
        self._counter += 1
        ts = int(time.time())
        safe_prefix = re.sub(r"[^\w]", "_", question[:30])
        question_id = f"question:{safe_prefix}_{ts}_{self._counter}"

        props = {
            "question": question,
            "category": category,
            "context": context,
            "status": "pending",
            "answer": "",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "related_node_id": related_node_id,
        }
        if extra_props:
            props.update(extra_props)

        # 写入 Question 节点
        self.gm.add_node(question_id, QUESTION_LABEL, props)

        # 写入 RAISES_QUESTION 边：从源节点指向 Question
        if self.gm.has_node(related_node_id):
            self.gm.add_edge(
                related_node_id,
                question_id,
                RAISES_QUESTION_EDGE,
                {"category": category},
            )
        else:
            logger.warning(
                f"[QuestionManager] 关联节点 '{related_node_id}' 不存在, "
                f"Question 已创建但未建立边"
            )

        logger.info(
            f"[QuestionManager] 创建问题 [{category}]: {question[:80]}..."
        )
        return question_id

    # ========================================================================
    # 查询
    # ========================================================================

    def list_pending(
        self,
        category: Optional[str] = None,
        skip: int = 0,
        limit: int = 50,
        keyword: Optional[str] = None,
    ) -> tuple:
        """
        列出待确认的问题。

        Returns:
            (items, total) — 分页后的列表 + 总匹配数
        """
        all_questions = self.gm.find_nodes_by_label(QUESTION_LABEL)
        results = []
        for q in all_questions:
            if q.get("status") != "pending":
                continue
            if category and q.get("category") != category:
                continue

            # Keyword filter
            if keyword:
                k_lower = keyword.lower()
                if k_lower not in q.get("question", "").lower() and k_lower not in q.get("context", "").lower():
                    continue

            results.append(q)

        # proposal 最优先，contradictory 次之
        priority = {"proposal": -1, "contradictory": 0, "untraceable": 1, "ambiguous": 2, "wrong_association": 3, "weak_association": 4, "game_trivia": 5}
        results.sort(key=lambda x: priority.get(x.get("category", ""), 9))

        total = len(results)
        return results[skip : skip + limit], total

    def list_all(
        self,
        status: Optional[str] = None,
        category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """列出所有问题（可按状态/分类过滤）。"""
        all_questions = self.gm.find_nodes_by_label(QUESTION_LABEL)
        results = []
        for q in all_questions:
            if status and q.get("status") != status:
                continue
            if category and q.get("category") != category:
                continue
            results.append(q)
        return results

    def get_questions_for_node(
        self,
        node_id: str,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """获取某个节点关联的所有问题。"""
        related = self.gm.find_related_nodes(
            node_id, relationship=RAISES_QUESTION_EDGE, direction="OUT"
        )
        results = []
        for r in related:
            if r.get("label") != QUESTION_LABEL:
                continue
            if status and r.get("status") != status:
                continue
            results.append(r)
        return results

    # ========================================================================
    # 回答 / 更新
    # ========================================================================

    def answer_question(self, question_id: str, answer: str) -> bool:
        """
        回答一个问题。

        Args:
            question_id: Question 节点 ID
            answer:      人类的回答

        Returns:
            是否成功
        """
        if not self.gm.has_node(question_id):
            logger.error(f"[QuestionManager] 问题 '{question_id}' 不存在")
            return False

        node_data = self.gm.get_node(question_id)
        if not node_data:
            return False
        node_data["answer"] = answer
        node_data["status"] = "answered"
        node_data["answered_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

        # 更新节点（通过 add_node 的 REPLACE 语义）
        label = node_data.pop("label", QUESTION_LABEL)
        node_data.pop("id", None)
        self.gm.add_node(question_id, label, node_data)

        logger.info(f"[QuestionManager] 问题已回答: {question_id}")
        return True

    def dismiss_question(self, question_id: str, reason: str = "") -> bool:
        """将问题标记为已忽略。"""
        if not self.gm.has_node(question_id):
            return False

        node_data = self.gm.get_node(question_id)
        if not node_data:
            return False
        node_data["status"] = "dismissed"
        node_data["dismiss_reason"] = reason
        node_data["dismissed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

        label = node_data.pop("label", QUESTION_LABEL)
        node_data.pop("id", None)
        self.gm.add_node(question_id, label, node_data)
        return True

    def approve_question(self, question_id: str) -> bool:
        """
        批准一个问题（用于 Game Trivia 审批流）。
        将状态设置为 'approved'。
        """
        if not self.gm.has_node(question_id):
            logger.error(f"[QuestionManager] 问题 '{question_id}' 不存在")
            return False

        node_data = self.gm.get_node(question_id)
        if not node_data:
            return False
        node_data["status"] = "approved"
        node_data["approved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

        label = node_data.pop("label", QUESTION_LABEL)
        node_data.pop("id", None)
        self.gm.add_node(question_id, label, node_data)

        logger.info(f"[QuestionManager] 问题已批准: {question_id}")
        return True

    def reject_question(self, question_id: str, reason: str = "") -> bool:
        """
        拒绝一个问题（用于 Game Trivia 审批流）。
        将状态设置为 'rejected'。
        """
        if not self.gm.has_node(question_id):
            logger.error(f"[QuestionManager] 问题 '{question_id}' 不存在")
            return False

        node_data = self.gm.get_node(question_id)
        if not node_data:
            return False
        node_data["status"] = "rejected"
        node_data["rejection_reason"] = reason
        node_data["rejected_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

        label = node_data.pop("label", QUESTION_LABEL)
        node_data.pop("id", None)
        self.gm.add_node(question_id, label, node_data)

        logger.info(f"[QuestionManager] 问题已拒绝: {question_id}")
        return True

    # ========================================================================
    # 统计
    # ========================================================================

    def get_stats(self) -> Dict[str, int]:
        """获取问题统计。"""
        all_q = self.gm.find_nodes_by_label(QUESTION_LABEL)
        stats = {"total": len(all_q), "pending": 0, "answered": 0, "dismissed": 0}
        by_category = {}
        for q in all_q:
            s = q.get("status", "pending")
            stats[s] = stats.get(s, 0) + 1
            c = q.get("category", "unknown")
            by_category[c] = by_category.get(c, 0) + 1

        stats["by_category"] = by_category
        return stats

    # ========================================================================
    # CLI 辅助
    # ========================================================================

    def format_question_for_display(self, q: Dict[str, Any]) -> str:
        """将 Question 节点格式化为人类可读的字符串。"""
        qid = q.get("id", "?")
        category = q.get("category", "?")
        status = q.get("status", "?")
        question = q.get("question", "?")
        context = q.get("context", "")
        related = q.get("related_node_id", "")
        answer = q.get("answer", "")
        created = q.get("created_at", "")

        lines = [
            f"[{category.upper()}] {question}",
            f"  ID:      {qid}",
            f"  Status:  {status}",
            f"  Related: {related}",
            f"  Context: {context[:200]}",
            f"  Created: {created}",
        ]
        if answer:
            lines.append(f"  Answer:  {answer}")
        return "\n".join(lines)
