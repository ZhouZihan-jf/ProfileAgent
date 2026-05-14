"""
Agent Graph - LangGraph 状态图组装

工作流:
    parse_tmf → load_config → rag_decision ──┬── need_rag=true  → rag_query → generate_profile
                                              └── need_rag=false → generate_profile
                                                                                 ↓
                                                                         validate_profile → END

LLM 实例传入需要模型推理的节点 (rag_decision, generate_profile)。
"""
import logging
from typing import Literal

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI

from src.agent.state import AgentState
from src.agent.nodes import parse_tmf, load_config, rag_decision, rag_query, generate_profile, \
    validate_profile

logger = logging.getLogger(__name__)


def _rag_decision_wrapper(llm: ChatOpenAI):
    """闭包注入 LLM 到 rag_decision 节点"""
    def inner(state: AgentState):
        return rag_decision(state, llm)
    return inner


def _generate_profile_wrapper(llm: ChatOpenAI):
    """闭包注入 LLM 到 generate_profile 节点"""
    def inner(state: AgentState):
        return generate_profile(state, llm)
    return inner


def create_graph(llm: ChatOpenAI) -> StateGraph:
    """
    构建 LangGraph 状态图

    Args:
        llm: ChatOpenAI 实例，注入到需要推理的节点
    """
    workflow = StateGraph(AgentState)

    # ========== 注册节点 ==========
    workflow.add_node("parse_tmf", parse_tmf)
    workflow.add_node("load_config", load_config)
    workflow.add_node("rag_decision", _rag_decision_wrapper(llm))
    workflow.add_node("rag_query", rag_query)
    workflow.add_node("generate_profile", _generate_profile_wrapper(llm))
    workflow.add_node("validate_profile", validate_profile)

    # ========== 定义边 ==========
    workflow.set_entry_point("parse_tmf")
    workflow.add_edge("parse_tmf", "load_config")
    workflow.add_edge("load_config", "rag_decision")

    # 条件边：rag_decision 根据 need_rag 决定下一步
    workflow.add_conditional_edges(
        "rag_decision",
        _route_after_decision,
        {
            "rag_query": "rag_query",
            "generate_profile": "generate_profile",
        },
    )

    workflow.add_edge("rag_query", "generate_profile")
    workflow.add_edge("generate_profile", "validate_profile")
    workflow.add_edge("validate_profile", END)

    return workflow


def _route_after_decision(state: AgentState) -> Literal["rag_query", "generate_profile"]:
    """根据 rag_decision 的结果决定下一步"""
    need_rag = state.get("need_rag", False)
    if need_rag:
        logger.info("→ 路由到 rag_query (需要 RAG 检索)")
        return "rag_query"
    else:
        logger.info("→ 路由到 generate_profile (跳过 RAG)")
        return "generate_profile"


def compile_agent(llm: ChatOpenAI, checkpointer=None):
    """
    编译并返回可执行的 Agent

    Args:
        llm: ChatOpenAI 实例
        checkpointer: 可选的自定义 checkpointer，默认使用 MemorySaver
    """
    workflow = create_graph(llm)
    if checkpointer is None:
        checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)
