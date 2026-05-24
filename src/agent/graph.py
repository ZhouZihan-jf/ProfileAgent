"""
Agent Graph - LangGraph 状态图组装 (v2 — 多 Plan 并行架构)

工作流:
    Phase 1: parse_tmf → slim_json → load_config
    Phase 2: generate_overview (LLM: 总览 + RAG 查询)
    Phase 3: rag_query (子 agent: 知识检索)
    Phase 4: dispatch → [plan_worker × N]  (Send 并行 Map-Reduce)
    Phase 5: generate_final_profile → validate_profile → END

无 Plan 时: Phase 4 自动跳过 → 直接 Phase 5
无 RAG 需求时: Phase 3 自动跳过 → 直接 Phase 4

LLM 实例通过闭包注入到需要模型推理的节点。
"""
import logging
from typing import Any, Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Send
from langchain_openai import ChatOpenAI

from src.agent.state import AgentState
from src.agent.nodes import (
    parse_tmf,
    slim_json,
    load_config,
    generate_overview,
    rag_query,
    plan_worker,
    generate_final_profile,
    validate_profile,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
#  闭包注入：LLM → nodes
# ═══════════════════════════════════════════════════════════════

def _overview_wrapper(llm: ChatOpenAI):
    def inner(state: AgentState):
        return generate_overview(state, llm)
    return inner


def _plan_worker_wrapper(llm: ChatOpenAI):
    def inner(state: AgentState):
        return plan_worker(state, llm)
    return inner


def _final_wrapper(llm: ChatOpenAI):
    def inner(state: AgentState):
        return generate_final_profile(state, llm)
    return inner


def _passthrough(_: AgentState) -> dict[str, Any]:
    """路由枢纽节点：不改变状态，仅为 Send fan-out 提供挂载点"""
    return {}


# ═══════════════════════════════════════════════════════════════
#  条件路由
# ═══════════════════════════════════════════════════════════════

def _route_after_overview(state: AgentState) -> Literal["rag_query", "dispatch_plans"]:
    """总览生成后：需要 RAG 则先去检索，否则直接分发 plans"""
    if state.get("need_rag", False) and state.get("rag_queries"):
        logger.info("→ 路由到 rag_query (需要 RAG 检索)")
        return "rag_query"
    logger.info("→ 路由到 dispatch_plans (跳过 RAG)")
    return "dispatch_plans"


def _fan_out_plans(state: AgentState):
    """
    Fan-out 路由：有 plans → Send[], 无 plans → 直接跳到汇总

    返回:
        - str "generate_final_profile": 无 plan，跳过并行处理
        - list[Send]: 每个 plan 一个 Send → plan_worker
    """
    plans = state.get("plans", [])
    if not plans:
        logger.info("plan_list 为空，跳过并行 → generate_final_profile")
        return "generate_final_profile"

    logger.info(f"Fan-out: {len(plans)} 个 plan → plan_worker (并行)")
    return [
        Send("plan_worker", {"plan_index": i})
        for i in range(len(plans))
    ]


# ═══════════════════════════════════════════════════════════════
#  图构建
# ═══════════════════════════════════════════════════════════════

def create_graph(llm: ChatOpenAI) -> StateGraph:
    """
    构建 LangGraph 状态图 (v2)

    Args:
        llm: ChatOpenAI 实例，注入到需要推理的节点
    """
    workflow = StateGraph(AgentState)

    # ===== 注册节点 =====
    workflow.add_node("parse_tmf", parse_tmf)
    workflow.add_node("slim_json", slim_json)
    workflow.add_node("load_config", load_config)  # type: ignore[arg-type]
    workflow.add_node("generate_overview", _overview_wrapper(llm))
    workflow.add_node("rag_query", rag_query)
    workflow.add_node("dispatch_plans", _passthrough)  # type: ignore[arg-type]   # 路由枢纽（无状态变更）
    workflow.add_node("plan_worker", _plan_worker_wrapper(llm))
    workflow.add_node("generate_final_profile", _final_wrapper(llm))
    workflow.add_node("validate_profile", validate_profile)

    # ===== Phase 1: 顺序预处理 =====
    workflow.set_entry_point("parse_tmf")
    workflow.add_edge("parse_tmf", "slim_json")
    workflow.add_edge("slim_json", "load_config")

    # ===== Phase 2 → RAG 条件路由 =====
    workflow.add_edge("load_config", "generate_overview")
    workflow.add_conditional_edges(
        "generate_overview",
        _route_after_overview,
        {"rag_query": "rag_query", "dispatch_plans": "dispatch_plans"},
    )

    # ===== Phase 3 → Phase 4 =====
    workflow.add_edge("rag_query", "dispatch_plans")

    # ===== Phase 4: Send Map-Reduce =====
    # dispatch_plans 完成 → _fan_out_plans 决定 Send[] 或直接跳到汇总
    #   - Send["plan_worker", ...] × N → 并行运行所有 worker
    #   - "generate_final_profile" → 无 plan 时直接汇总
    workflow.add_conditional_edges(
        "dispatch_plans",
        _fan_out_plans,
        {"plan_worker": "plan_worker", "generate_final_profile": "generate_final_profile"},
    )
    # 所有 plan_worker 的 Send 实例完成后 → 汇总
    workflow.add_edge("plan_worker", "generate_final_profile")

    # ===== Phase 5: 汇总 → 校验 → END =====
    workflow.add_edge("generate_final_profile", "validate_profile")
    workflow.add_edge("validate_profile", END)

    return workflow


def compile_agent(llm: ChatOpenAI, checkpointer=None):
    """
    编译并返回可执行的 Agent

    Args:
        llm: ChatOpenAI 实例
        checkpointer: 可选的自定义 checkpointer，默认 MemorySaver
    """
    workflow = create_graph(llm)
    if checkpointer is None:
        checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)
