"""
AgentState - LangGraph 核心状态定义 (v2 — 多 Plan 并行架构)

贯穿整个工作流的共享状态。plan_results 通过自定义 reducer 在 Send 并行分支
收敛时自动合并，无需手动聚合。
"""

import operator
from typing import Annotated, Any, Optional, Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


def _merge_plan_results(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Send 并行分支收敛时自动合并 plan_results"""
    return left + right


class AgentState(TypedDict):
    """Agent 工作流状态 (v2 — 支持 plan 级并行)"""

    # ========== 输入 (Phase 1 产出) ==========
    tmf_input: dict[str, Any]
    """原始 TMF 格式的套餐 offer JSON（完整）"""

    tmf_basic_info: dict[str, Any]
    """瘦身后的 offer 基本信息（不含 plans，只有 top-level 字段）"""

    plans: list[dict[str, Any]]
    """瘦身后的 plan 列表，每个 plan 只保留模型需要的关键字段"""

    # ========== 配置 (Phase 1 产出) ==========
    rules_md: str
    """RAG API 使用规则 (Markdown 内容)"""

    profile_template: dict[str, Any]
    """画像输出模板"""

    rag_api_config: dict[str, Any]
    """RAG API 连接配置"""

    # ========== 消息流 ==========
    messages: Annotated[Sequence[BaseMessage], add_messages]
    """LLM 对话消息历史"""

    # ========== Phase 2: 总览 (LLM 产出) ==========
    offer_overview: str
    """LLM 生成的 offer 总览（包含整体定位、目标客群、核心卖点等）"""

    need_rag: bool
    """是否需要调用 RAG"""

    rag_queries: list[str]
    """LLM 生成的 RAG 检索查询列表"""

    # ========== Phase 3: RAG ==========
    rag_context: list[dict[str, Any]]
    """RAG 返回的上下文文档"""

    rag_retry_count: int
    """RAG 重试次数（0-based，max 由配置决定）"""

    rag_sufficiency_report: Optional[dict[str, Any]]
    """RAG 检索充分性评估报告"""

    plan_rag_contexts: dict[int, list[dict[str, Any]]]
    """Plan 级差异化 RAG 上下文: {plan_index: [docs]}"""

    # ========== Phase 4: 并行 Plan 处理 ==========
    plan_index: int
    """当前 worker 处理的 plan 索引（由 Send 注入，0-based）"""

    plan_results: Annotated[list[dict[str, Any]], _merge_plan_results]
    """各 plan_worker 产出的画像片段，Send 收敛时自动合并"""

    # ========== Phase 5: 输出 ==========
    profile_output: Optional[dict[str, Any]]
    """生成的 offer 画像 JSON（汇总后）"""

    validation_errors: list[str]
    """校验阶段发现的错误"""

    final_output: Optional[dict[str, Any]]
    """最终输出的画像 JSON"""
