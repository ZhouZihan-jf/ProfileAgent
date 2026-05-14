"""
AgentState - LangGraph 核心状态定义

贯穿整个工作流的共享状态，各节点通过读写 State 协作完成画像生成。
"""

from typing import Annotated, Any, Optional, Sequence
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """Agent 工作流状态"""

    # ========== 输入 ==========
    tmf_input: dict[str, Any]
    """原始 TMF 格式的运营商套餐 offer JSON"""

    # ========== 配置 ==========
    rules_md: str
    """RAG API 使用规则 (Markdown 内容)"""

    profile_template: dict[str, Any]
    """画像输出模板 (可配置的 JSON Schema)"""

    rag_api_config: dict[str, Any]
    """RAG API 连接配置 (base_url, endpoints, auth 等)"""

    # ========== 消息流 ==========
    messages: Annotated[Sequence[BaseMessage], add_messages]
    """LLM 对话消息历史"""

    # ========== RAG 相关 ==========
    need_rag: bool
    """模型自主决策：是否需要调用 RAG"""

    rag_queries: list[str]
    """模型生成的 RAG 查询列表"""

    rag_context: list[dict[str, Any]]
    """RAG 返回的上下文文档"""

    # ========== 输出 ==========
    profile_output: Optional[dict[str, Any]]
    """生成的 offer 画像 JSON"""

    validation_errors: list[str]
    """校验阶段发现的错误"""

    final_output: Optional[dict[str, Any]]
    """最终输出的画像 JSON"""
