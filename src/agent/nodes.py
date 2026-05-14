"""
Agent 工作流节点 — 6 个节点函数构成完整 Pipeline

流程: parse_tmf → load_config → rag_decision ─┬─ rag_query ──→ generate_profile → validate_profile
                                               └─ (skip) ───→ generate_profile
"""

import json
import logging
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

from src.agent.state import AgentState
from src.config.config import load_prompt, load_profile_template, load_rag_config, load_rules, \
    validate_tmf_input, RAGAPIConfig
from src.rag.client import RAGClient

logger = logging.getLogger(__name__)


# ==============================================================
#  Node 1: parse_tmf
# ==============================================================

def parse_tmf(state: AgentState) -> dict[str, Any]:
    """
    解析并校验 TMF 输入 JSON

    处理：
        1. 如果是字符串，尝试 JSON 解析
        2. 校验必填字段 (id, name)
        3. 返回校验后的数据
    """
    tmf_data = state["tmf_input"]
    if isinstance(tmf_data, str):
        tmf_data = json.loads(tmf_data)

    validated = validate_tmf_input(tmf_data)
    return {"tmf_input": validated}


# ==============================================================
#  Node 2: load_config
# ==============================================================

def load_config(state: AgentState) -> dict[str, Any]:
    """
    加载运行时配置：画像模板、RAG 规则、RAG API 配置
    """
    try:
        template = load_profile_template()
        logger.info(f"加载画像模板: {template.name} v{template.version}")
        template_dict = template.model_dump()
    except FileNotFoundError:
        logger.warning("画像模板文件不存在，使用最小默认模板")
        template_dict = _get_fallback_template()

    try:
        rules = load_rules()
        logger.info(f"加载 RAG 规则 ({len(rules)} 字符)")
    except FileNotFoundError:
        logger.warning("RAG 规则文件不存在")
        rules = ""

    try:
        rag_config = load_rag_config()
        logger.info(f"加载 RAG API 配置: {rag_config.base_url}")
        rag_dict = rag_config.model_dump()
    except FileNotFoundError:
        logger.warning("RAG API 配置不存在，RAG 功能将禁用")
        rag_dict = {"base_url": "", "endpoints": [], "api_key": None}

    return {
        "profile_template": template_dict,
        "rules_md": rules,
        "rag_api_config": rag_dict,
    }


def _get_fallback_template() -> dict[str, Any]:
    """默认兜底画像模板"""
    return {
        "name": "default",
        "version": "1.0.0",
        "fields": [
            {"name": "offer_id", "type": "string", "required": True,
             "description": "套餐ID", "source": "tmf"},
            {"name": "offer_name", "type": "string", "required": True,
             "description": "套餐名称", "source": "tmf"},
            {"name": "target_users", "type": "array", "required": True,
             "description": "目标用户群画像列表", "source": "model"},
            {"name": "key_features", "type": "array", "required": True,
             "description": "核心卖点", "source": "model"},
            {"name": "competitive_advantages", "type": "array", "required": False,
             "description": "竞争优势", "source": "rag"},
            {"name": "tags", "type": "array", "required": False,
             "description": "标签分类", "source": "model"},
        ],
    }


# ==============================================================
#  Node 3: rag_decision
# ==============================================================

_DECISION_PROMPT_TEMPLATE: str | None = None


def _get_decision_prompt() -> str:
    global _DECISION_PROMPT_TEMPLATE
    if _DECISION_PROMPT_TEMPLATE is None:
        _DECISION_PROMPT_TEMPLATE = load_prompt("rag_decision")
    return _DECISION_PROMPT_TEMPLATE


def rag_decision(state: AgentState, llm: ChatOpenAI) -> dict[str, Any]:
    """
    模型自主决策是否调用 RAG

    依据 rules.md 规则、模板中 source=rag 的字段、TMF 输入信息，
    判断是否需要调用 RAG API，如需则生成检索查询。
    """
    rules_md = state.get("rules_md", "")
    tmf_input = state.get("tmf_input", {})
    template = state.get("profile_template", {})

    # 提取需要 RAG 的字段，若无则直接跳过
    rag_fields = [f for f in template.get("fields", []) if f.get("source") == "rag"]
    if not rag_fields:
        logger.info("模板中无 source=rag 字段，跳过 RAG")
        return {"need_rag": False, "rag_queries": []}

    # 构建 prompt
    tmf_summary = json.dumps({
        "name": tmf_input.get("name", ""),
        "description": tmf_input.get("description", ""),
        "category": tmf_input.get("category", ""),
        "characteristics": tmf_input.get("characteristics", {}),
    }, ensure_ascii=False, indent=2)

    template_fields = json.dumps(
        [{"name": f["name"], "description": f.get("description", ""), "source": f.get("source", "")}
         for f in template.get("fields", [])],
        ensure_ascii=False, indent=2,
    )

    prompt = _get_decision_prompt().format(
        rules=rules_md if rules_md else "（未配置规则，根据字段需求自行判断）",
        tmf_summary=tmf_summary,
        template_fields=template_fields,
    )

    try:
        response = llm.invoke([
            SystemMessage(content="你只返回 JSON，不要任何额外文本。"),
            HumanMessage(content=prompt),
        ])
        result = _parse_decision(response.content)
        logger.info(f"RAG 决策: need_rag={result['need_rag']}, queries={len(result.get('queries', []))}个")
        return {
            "need_rag": result["need_rag"],
            "rag_queries": result.get("queries", []),
        }
    except Exception as e:
        logger.warning(f"RAG 决策解析失败: {e}，默认不调用 RAG")
        return {"need_rag": False, "rag_queries": []}


def _parse_decision(raw: str) -> dict[str, Any]:
    """解析模型输出的决策 JSON"""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return json.loads(text)


# ==============================================================
#  Node 4: rag_query
# ==============================================================

def rag_query(state: AgentState) -> dict[str, Any]:
    """
    调用 RAG API 检索知识

    从 rag_api_config 创建 RAGClient，批量检索 rag_queries，
    将结果整理后存入 rag_context。
    """
    rag_cfg: dict[str, Any] = state["rag_api_config"]  # type: ignore[assignment]
    queries: list[str] = state["rag_queries"]  # type: ignore[assignment]

    if not queries:
        logger.info("无 RAG 查询，跳过")
        return {"rag_context": []}

    if not rag_cfg.get("base_url"):
        logger.warning("RAG API 未配置 base_url，跳过查询")
        return {
            "rag_context": [],
            "messages": [AIMessage(content="[RAG] API 未配置 base_url，跳过知识检索。")],
        }

    try:
        config = RAGAPIConfig(
            base_url=rag_cfg.get("base_url", ""),  # type: ignore[reportAny]
            api_key=rag_cfg.get("api_key"),  # type: ignore[reportAny]
            endpoints=rag_cfg.get("endpoints", []),  # type: ignore[reportAny]
            timeout=rag_cfg.get("timeout", 30),  # type: ignore[reportAny]
            max_retries=rag_cfg.get("max_retries", 2),  # type: ignore[reportAny]
            default_top_k=rag_cfg.get("default_top_k", 5),  # type: ignore[reportAny]
        )
        client = RAGClient(config)
        results = client.batch_search(queries)
        client.close()
    except Exception as e:
        logger.error(f"RAG 客户端初始化失败: {e}")
        return {
            "rag_context": [],
            "messages": [AIMessage(content=f"[RAG] 连接失败: {e}")],
        }

    context_list: list[dict[str, Any]] = []
    for query, docs in results.items():
        for doc in docs[:3]:
            context_list.append({
                "query": query,
                "content": doc.get("content", ""),
                "score": doc.get("score", 0),
                "metadata": doc.get("metadata", {}),
            })

    logger.info(f"RAG 检索完成: {len(context_list)} 条文档 (来自 {len(queries)} 个查询)")
    return {
        "rag_context": context_list,
        "messages": [AIMessage(content=f"[RAG] 检索到 {len(context_list)} 条相关文档")],
    }


# ==============================================================
#  Node 5: generate_profile
# ==============================================================

_GENERATION_PROMPT_TEMPLATE: str | None = None


def _get_generation_prompt() -> str:
    global _GENERATION_PROMPT_TEMPLATE
    if _GENERATION_PROMPT_TEMPLATE is None:
        _GENERATION_PROMPT_TEMPLATE = load_prompt("generate_profile")
    return _GENERATION_PROMPT_TEMPLATE


def generate_profile(state: AgentState, llm: ChatOpenAI) -> dict[str, Any]:
    """
    根据模板生成 offer 画像 JSON

    融合 TMF 输入、RAG 检索上下文和画像模板，由 LLM 生成结构化画像。
    """
    template = state.get("profile_template", {})
    tmf_input = state.get("tmf_input", {})
    rag_context = state.get("rag_context", [])

    template_json = json.dumps(
        {"fields": template.get("fields", [])},
        ensure_ascii=False, indent=2,
    )
    tmf_json = json.dumps(tmf_input, ensure_ascii=False, indent=2)

    rag_text = "（无 RAG 检索结果）"
    if rag_context:
        items = []
        for i, ctx in enumerate(rag_context, 1):
            items.append(f"[{i}] 查询: {ctx['query']}\n    内容: {ctx['content'][:500]}")
        rag_text = "\n".join(items)

    prompt = _get_generation_prompt().format(
        template_json=template_json,
        tmf_json=tmf_json,
        rag_context=rag_text,
    )

    try:
        response = llm.invoke([
            SystemMessage(content="你只返回 JSON，严格按字段定义结构，不要额外文本。"),
            HumanMessage(content=prompt),
        ])
        profile = _parse_profile_response(response.content)
        logger.info(f"画像生成完成，包含 {len(profile)} 个顶层字段")
        return {"profile_output": profile}
    except Exception as e:
        logger.error(f"画像生成失败: {e}")
        return {
            "profile_output": {"error": str(e)},
            "messages": [SystemMessage(content=f"[Profile] 生成失败: {e}")],
        }


def _parse_profile_response(raw: str) -> dict[str, Any]:
    """解析 LLM 输出的画像 JSON"""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return json.loads(text)


# ==============================================================
#  Node 6: validate_profile
# ==============================================================

def validate_profile(state: AgentState) -> dict[str, Any]:
    """
    校验生成的画像 JSON

    检查项：
    1. required 字段是否存在且不为空
    2. 字段类型匹配
    3. 数据完整性
    """
    template = state.get("profile_template", {})
    profile = state.get("profile_output")
    errors: list[str] = []

    if profile is None:
        return {"validation_errors": ["profile_output 为空"], "final_output": None}

    if "error" in profile:
        return {
            "validation_errors": [f"生成阶段出错: {profile['error']}"],
            "final_output": profile,
        }

    fields = template.get("fields", [])

    for field in fields:
        name = field.get("name", "")
        required = field.get("required", False)
        ftype = field.get("type", "string")
        value = profile.get(name)

        if required:
            if value is None:
                errors.append(f"缺少必填字段: {name}")
                continue
            if isinstance(value, (list, str, dict)) and len(value) == 0:
                errors.append(f"必填字段为空: {name}")

        if value is not None:
            if not _check_type(value, ftype):
                errors.append(f"字段类型错误: {name} 期望 {ftype}, 实际 {type(value).__name__}")

    if errors:
        logger.warning(f"画像校验发现 {len(errors)} 个问题")
        for err in errors:
            logger.warning(f"  - {err}")
        return {"validation_errors": errors, "final_output": profile}

    logger.info("画像校验通过")
    return {"validation_errors": [], "final_output": profile}


def _check_type(value: Any, expected: str) -> bool:
    """简单类型校验"""
    type_map = {
        "string": str,
        "number": (int, float),
        "integer": int,
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    expected_types = type_map.get(expected)
    if expected_types is None:
        return True
    return isinstance(value, expected_types)
