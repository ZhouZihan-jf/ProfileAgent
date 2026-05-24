"""
Agent 工作流节点 (v2 — 多 Plan 并行架构)

流程:
  Phase 1: parse_tmf → slim_json → load_config
  Phase 2: generate_overview (LLM: 生成总览 + RAG 查询)
  Phase 3: rag_query (子 agent: 执行知识检索)
  Phase 4: dispatch_plans → [plan_worker × N]  (Send 并行)
  Phase 5: generate_final_profile → validate_profile
"""

import json
import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.agent.state import AgentState
from src.config.config import (
    RAGAPIConfig,
    load_prompt,
    load_profile_template,
    load_rag_config,
    load_rules,
    validate_tmf_input,
)
from src.rag.client import RAGClient

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════
#  Phase 1: 预处理 (纯代码，无 LLM 调用)
# ═════════════════════════════════════════════════════════════════

# ----- Slim 配置：各层级保留的关键字段 -----
_TOP_LEVEL_FIELDS = {"id", "name", "description", "category", "characteristics",
                      "product_specification", "bundled_product_offering"}

_PLAN_FIELDS = {"id", "name", "description", "type", "status", "lifecycle_status"}

_POLICY_FIELDS = {"id", "name", "description", "type", "priority", "status"}

_PATTERN_FIELDS = {"id", "name", "description", "type"}

_ACTION_FIELDS = {"id", "name", "description", "type", "value", "unit"}


def _slim_dict(data: dict[str, Any], keep_fields: set[str]) -> dict[str, Any]:
    """只保留指定字段，递归处理嵌套 dict"""
    slimmed: dict[str, Any] = {}
    for key, value in data.items():
        if key not in keep_fields:
            continue
        if isinstance(value, dict):
            slimmed[key] = _slim_dict(value, keep_fields)
        elif isinstance(value, list):
            slimmed[key] = [
                _slim_dict(item, keep_fields) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            slimmed[key] = value
    return slimmed


def _slim_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """对单个 plan 递归瘦身：plan → policy → pattern → action"""
    slimmed: dict[str, Any] = {}

    # 保留 plan 级别字段
    for key in plan:
        if key in _PLAN_FIELDS:
            slimmed[key] = plan[key]

    # 瘦身 policy
    policy = plan.get("policy")
    if isinstance(policy, dict):
        slimmed_policy: dict[str, Any] = {}
        for key in policy:
            if key in _POLICY_FIELDS:
                slimmed_policy[key] = policy[key]

        # 瘦身 pattern (在 policy 内部)
        pattern = policy.get("pattern")
        if isinstance(pattern, dict):
            slimmed_pattern: dict[str, Any] = {}
            for key in pattern:
                if key in _PATTERN_FIELDS:
                    slimmed_pattern[key] = pattern[key]

            # 瘦身 action (在 pattern 内部)
            action = pattern.get("action")
            if isinstance(action, dict):
                slimmed_pattern["action"] = _slim_dict(action, _ACTION_FIELDS)
            elif isinstance(action, list):
                slimmed_pattern["action"] = [
                    _slim_dict(a, _ACTION_FIELDS) if isinstance(a, dict) else a
                    for a in action
                ]

            slimmed_policy["pattern"] = slimmed_pattern

        slimmed["policy"] = slimmed_policy

    # 嵌套 plans（递归）
    nested_plans = plan.get("plans")
    if isinstance(nested_plans, list):
        slimmed["plans"] = [_slim_plan(p) for p in nested_plans]

    return slimmed


def parse_tmf(state: AgentState) -> dict[str, Any]:
    """
    解析并校验 TMF 输入 JSON

    1. 字符串 → dict
    2. 校验必填字段 (id, name)
    3. 确保 plan_list 存在
    """
    tmf_data = state["tmf_input"]
    if isinstance(tmf_data, str):
        tmf_data = json.loads(tmf_data)

    validated = validate_tmf_input(tmf_data)

    # 确保 plan_list 存在
    if "plan_list" not in validated:
        logger.warning("TMF 输入中未找到 plan_list，使用空列表")
        validated["plan_list"] = []

    plan_count = len(validated.get("plan_list", []))
    logger.info(f"TMF 校验通过: id={validated['id']}, plans={plan_count}")
    return {"tmf_input": validated}


def slim_json(state: AgentState) -> dict[str, Any]:
    """
    JSON 瘦身：提取基本信息 + 对 plan_list 逐项瘦身

    顶层只保留 id/name/description/category 等关键字段，
    每个 plan 递归裁剪到 plan→policy→pattern→action 的关键字段。
    """
    tmf_input = state["tmf_input"]

    # 1. 提取基本信息（不含 plan_list）
    basic_info = _slim_dict(tmf_input, _TOP_LEVEL_FIELDS)

    # 2. 瘦身所有 plans
    raw_plans = tmf_input.get("plan_list", [])
    slimmed_plans = [_slim_plan(p) for p in raw_plans if isinstance(p, dict)]

    # 3. 统计瘦身效果
    original_size = len(json.dumps(tmf_input, ensure_ascii=False))
    slimmed_size = len(json.dumps(
        {**basic_info, "plan_list": slimmed_plans}, ensure_ascii=False
    ))
    reduction = (1 - slimmed_size / max(original_size, 1)) * 100

    logger.info(
        f"JSON 瘦身完成: {original_size} → {slimmed_size} 字符 "
        f"({reduction:.0f}% 缩减), plans={len(slimmed_plans)}"
    )
    return {
        "tmf_basic_info": basic_info,
        "plans": slimmed_plans,
    }


def load_config(state: AgentState) -> dict[str, Any]:
    """加载运行时配置：画像模板、RAG 规则、RAG API 配置"""
    try:
        template = load_profile_template()
        template_dict = template.model_dump()
    except FileNotFoundError:
        logger.warning("画像模板文件不存在，使用最小默认模板")
        template_dict = _get_fallback_template()

    try:
        rules = load_rules()
    except FileNotFoundError:
        logger.warning("RAG 规则文件不存在")
        rules = ""

    try:
        rag_config = load_rag_config()
        rag_dict = rag_config.model_dump()
    except FileNotFoundError:
        logger.warning("RAG API 配置不存在，RAG 功能将禁用")
        rag_dict = {"base_url": "", "endpoints": [], "api_key": None}

    logger.info(f"配置加载完成: 模板={template_dict.get('name')}, "
                f"rules={len(rules)}字符, RAG={rag_dict.get('base_url', 'N/A')}")
    return {
        "profile_template": template_dict,
        "rules_md": rules,
        "rag_api_config": rag_dict,
    }


def _get_fallback_template() -> dict[str, Any]:
    return {
        "name": "default",
        "version": "2.0.0",
        "fields": [
            {"name": "offer_id", "type": "string", "required": True,
             "description": "套餐ID", "source": "tmf"},
            {"name": "offer_name", "type": "string", "required": True,
             "description": "套餐名称", "source": "tmf"},
            {"name": "offer_overview", "type": "string", "required": True,
             "description": "套餐总览", "source": "model"},
            {"name": "target_users", "type": "array", "required": True,
             "description": "目标用户群画像", "source": "model"},
            {"name": "key_features", "type": "array", "required": True,
             "description": "核心卖点", "source": "model"},
            {"name": "competitive_advantages", "type": "array", "required": False,
             "description": "竞争优势", "source": "rag"},
            {"name": "plan_profiles", "type": "array", "required": True,
             "description": "各Plan画像", "source": "model"},
            {"name": "market_insights", "type": "array", "required": False,
             "description": "市场趋势分析", "source": "rag"},
        ],
    }


# ═════════════════════════════════════════════════════════════════
#  Phase 2: 总览生成 (LLM × 1)
# ═════════════════════════════════════════════════════════════════

_OVERVIEW_PROMPT: str | None = None


def _get_overview_prompt() -> str:
    global _OVERVIEW_PROMPT
    if _OVERVIEW_PROMPT is None:
        _OVERVIEW_PROMPT = load_prompt("generate_overview")
    return _OVERVIEW_PROMPT


def generate_overview(state: AgentState, llm: ChatOpenAI) -> dict[str, Any]:
    """
    生成 offer 总览 + RAG 检索查询

    输入：瘦身后的 basic_info + plans 摘要 + 模板字段 + rules
    输出：JSON { overview, need_rag, queries, plan_summaries }
    """
    basic_info = state.get("tmf_basic_info", {})
    plans = state.get("plans", [])
    rules_md = state.get("rules_md", "")
    template = state.get("profile_template", {})

    # 构建 plans 摘要（只传 id + name + description，不传完整结构）
    plan_summaries = [
        {
            "index": i,
            "id": p.get("id", f"plan_{i}"),
            "name": p.get("name", ""),
            "description": p.get("description", ""),
        }
        for i, p in enumerate(plans)
    ]

    # 提取需要 RAG 的字段
    rag_fields = [f for f in template.get("fields", []) if f.get("source") == "rag"]
    template_fields_desc = json.dumps(
        [{"name": f["name"], "description": f.get("description", ""), "source": f.get("source", "")}
         for f in template.get("fields", [])],
        ensure_ascii=False, indent=2,
    )

    prompt = _get_overview_prompt().format(
        rules=rules_md if rules_md else "（未配置 RAG 规则）",
        offer_basic=json.dumps(basic_info, ensure_ascii=False, indent=2),
        plan_summaries=json.dumps(plan_summaries, ensure_ascii=False, indent=2),
        plan_count=len(plan_summaries),
        template_fields=template_fields_desc,
        rag_field_count=len(rag_fields),
    )

    try:
        response = llm.invoke([
            SystemMessage(content="你只返回 JSON，不要任何额外文本。"),
            HumanMessage(content=prompt),
        ])
        result = _parse_json_response(response.content)

        overview = result.get("overview", "")
        need_rag = result.get("need_rag", False)
        queries = result.get("queries", [])

        logger.info(
            f"总览生成完成: overview={len(overview)}字符, "
            f"need_rag={need_rag}, queries={len(queries)}"
        )
        return {
            "offer_overview": overview,
            "need_rag": need_rag,
            "rag_queries": queries,
        }
    except Exception as e:
        logger.warning(f"总览生成失败: {e}，使用降级方案")
        return {
            "offer_overview": basic_info.get("description", basic_info.get("name", "")),
            "need_rag": False,
            "rag_queries": [],
        }


# ═════════════════════════════════════════════════════════════════
#  Phase 2.5: 查询优化 (LLM × 1)
# ═════════════════════════════════════════════════════════════════

_QUERY_REFINE_PROMPT: str | None = None


def _get_query_refine_prompt() -> str:
    global _QUERY_REFINE_PROMPT
    if _QUERY_REFINE_PROMPT is None:
        _QUERY_REFINE_PROMPT = load_prompt("query_refine")
    return _QUERY_REFINE_PROMPT


def query_refine(state: AgentState, llm: ChatOpenAI) -> dict[str, Any]:
    """
    查询扩展优化：将 LLM 生成的原始查询扩展为多角度变体

    输入：rag_queries, template, basic_info
    输出：expanded_queries（替换原始 rag_queries）
    """
    queries = state.get("rag_queries", [])
    if not queries:
        logger.info("query_refine: 无查询，跳过")
        return {}

    template = state.get("profile_template", {})
    basic_info = state.get("tmf_basic_info", {})

    # 提取 source=rag 字段描述
    rag_fields = [
        {"name": f["name"], "description": f.get("description", ""),
         "source": f.get("source", "")}
        for f in template.get("fields", [])
        if f.get("source") == "rag"
    ]

    prompt = _get_query_refine_prompt().format(
        template_fields=json.dumps(rag_fields, ensure_ascii=False, indent=2),
        queries=json.dumps(queries, ensure_ascii=False, indent=2),
        offer_basic=json.dumps(
            {"id": basic_info.get("id"), "name": basic_info.get("name"),
             "description": basic_info.get("description", "")},
            ensure_ascii=False, indent=2,
        ),
    )

    try:
        response = llm.invoke([
            SystemMessage(content="你只返回 JSON，不要任何额外文本。"),
            HumanMessage(content=prompt),
        ])
        result = _parse_json_response(response.content)
        expanded = result.get("expanded_queries", queries)
        reasoning = result.get("reasoning", "")

        logger.info(
            f"查询优化完成: {len(queries)} → {len(expanded)} 条, "
            f"策略: {reasoning[:80]}"
        )
        return {"rag_queries": expanded}
    except Exception as e:
        logger.warning(f"查询优化失败: {e}，使用原始查询")
        return {}  # 保留原始 rag_queries


# ═════════════════════════════════════════════════════════════════
#  Phase 3: RAG 检索 (子 agent)
# ═════════════════════════════════════════════════════════════════

def rag_query(state: AgentState) -> dict[str, Any]:
    """
    执行 RAG 知识检索

    从 rag_api_config 创建 RAGClient，批量检索 rag_queries。
    need_rag=False 或无 queries 时跳过。
    """
    if not state.get("need_rag", False):
        logger.info("need_rag=False，跳过 RAG 检索")
        return {"rag_context": []}

    rag_cfg: dict[str, Any] = state["rag_api_config"]
    queries: list[str] = state.get("rag_queries", [])

    if not queries:
        logger.info("无 RAG 查询，跳过")
        return {"rag_context": []}

    if not rag_cfg.get("base_url"):
        logger.warning("RAG API 未配置 base_url，跳过检索")
        return {
            "rag_context": [],
            "messages": [AIMessage(content="[RAG] API 未配置，跳过知识检索。")],
        }

    try:
        min_score = rag_cfg.get("min_score", 0.3)
        config = RAGAPIConfig(
            base_url=rag_cfg.get("base_url", ""),
            api_key=rag_cfg.get("api_key"),
            endpoints=rag_cfg.get("endpoints", []),
            timeout=rag_cfg.get("timeout", 30),
            max_retries=rag_cfg.get("max_retries", 2),
            default_top_k=rag_cfg.get("default_top_k", 5),
            min_score=min_score,
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

    # 收集所有文档 + 评分过滤
    all_docs: list[dict[str, Any]] = []
    discarded_count = 0
    for query, docs in results.items():
        for doc in docs:
            score = doc.get("score", 0)
            if score < min_score:
                discarded_count += 1
                continue
            all_docs.append({
                "query": query,
                "content": doc.get("content", ""),
                "score": score,
                "metadata": doc.get("metadata", {}),
            })

    # 按评分降序排序，限制总数
    all_docs.sort(key=lambda x: x["score"], reverse=True)
    context_list = all_docs[:20]

    logger.info(
        f"RAG 检索完成: {len(context_list)} 条文档 "
        f"(过滤 {discarded_count} 条低分, 共 {len(queries)} 个查询)"
    )
    return {
        "rag_context": context_list,
        "messages": [AIMessage(content=f"[RAG] 检索到 {len(context_list)} 条相关文档")],
    }


# ═════════════════════════════════════════════════════════════════
#  Phase 3.5: 检索充分性检查 (LLM × 1)
# ═════════════════════════════════════════════════════════════════

_SUFFICIENCY_PROMPT: str | None = None


def _get_sufficiency_prompt() -> str:
    global _SUFFICIENCY_PROMPT
    if _SUFFICIENCY_PROMPT is None:
        _SUFFICIENCY_PROMPT = load_prompt("rag_sufficiency_check")
    return _SUFFICIENCY_PROMPT


def rag_sufficiency_check(state: AgentState, llm: ChatOpenAI) -> dict[str, Any]:
    """
    评估 RAG 检索结果是否足够支撑画像生成

    检查每个 source=rag 字段的覆盖度，不满足时生成修正查询供重试。
    同时自增 rag_retry_count。
    """
    rag_context: list[dict[str, Any]] = state.get("rag_context", [])
    template = state.get("profile_template", {})
    retry_count = state.get("rag_retry_count", 0)

    if not rag_context:
        logger.info("rag_sufficiency_check: RAG 无结果，跳过检查")
        return {
            "rag_sufficiency_report": {
                "sufficient": True,
                "overall_confidence": 0.0,
                "summary": "无 RAG 结果，无需检查",
            },
            "rag_retry_count": retry_count + 1,
        }

    # 提取 source=rag 字段
    rag_fields = [
        {
            "name": f.get("name"),
            "description": f.get("description", ""),
            "source": f.get("source", ""),
        }
        for f in template.get("fields", [])
        if f.get("source") == "rag"
    ]

    # 构建文档摘要（只传关键信息给 LLM）
    doc_summaries = [
        {
            "idx": i,
            "query": d.get("query", ""),
            "content_preview": d.get("content", "")[:200],
            "score": d.get("score", 0),
        }
        for i, d in enumerate(rag_context)
    ]

    prompt = _get_sufficiency_prompt().format(
        template_fields=json.dumps(rag_fields, ensure_ascii=False, indent=2),
        queries=json.dumps(state.get("rag_queries", []), ensure_ascii=False),
        rag_docs=json.dumps(doc_summaries, ensure_ascii=False, indent=2),
        doc_count=len(rag_context),
        retry_count=retry_count,
    )

    try:
        response = llm.invoke([
            SystemMessage(content="你只返回 JSON，不要任何额外文本。"),
            HumanMessage(content=prompt),
        ])
        report = _parse_json_response(response.content)
        sufficient = report.get("sufficient", True)
        confidence = report.get("overall_confidence", 0)
        refined = report.get("refined_queries", [])

        logger.info(
            f"RAG 充分性检查: sufficient={sufficient}, "
            f"confidence={confidence:.2f}, "
            f"gaps={len(refined)}个修正查询"
        )

        result: dict[str, Any] = {
            "rag_sufficiency_report": report,
            "rag_retry_count": retry_count + 1,
        }

        # 不充分时更新 rag_queries 用于重试
        if not sufficient and refined:
            result["rag_queries"] = refined

        return result
    except Exception as e:
        logger.warning(f"RAG 充分性检查失败: {e}，假设充分")
        return {
            "rag_sufficiency_report": {
                "sufficient": True,
                "overall_confidence": 0.5,
                "summary": f"评估异常: {e}",
            },
            "rag_retry_count": retry_count + 1,
        }


# ═════════════════════════════════════════════════════════════════
#  Phase 3.8: Plan 级差异化 RAG (LLM × 1 + RAGClient × N)
# ═════════════════════════════════════════════════════════════════

_PLAN_RAG_PLANNER_PROMPT: str | None = None


def _get_plan_rag_planner_prompt() -> str:
    global _PLAN_RAG_PLANNER_PROMPT
    if _PLAN_RAG_PLANNER_PROMPT is None:
        _PLAN_RAG_PLANNER_PROMPT = load_prompt("plan_rag_planner")
    return _PLAN_RAG_PLANNER_PROMPT


def plan_rag_planner(state: AgentState, llm: ChatOpenAI) -> dict[str, Any]:
    """
    为每个 Plan 判断是否需要独立 RAG，并为需要的 Plan 执行检索

    输入：plans, rag_context(global), offer_overview, template
    输出：plan_rag_contexts = {plan_index: [docs]}
    """
    plans = state.get("plans", [])
    rag_context = state.get("rag_context", [])
    offer_overview = state.get("offer_overview", "")
    template = state.get("profile_template", {})
    rag_cfg: dict[str, Any] = state.get("rag_api_config", {})

    if not plans:
        logger.info("plan_rag_planner: 无 Plan，跳过")
        return {"plan_rag_contexts": {}}

    # 没有全局 RAG 且 RAG API 不可用 → 跳过
    if not rag_context and not rag_cfg.get("base_url"):
        logger.info("plan_rag_planner: 无 RAG 上下文且 API 不可用，跳过")
        return {"plan_rag_contexts": {}}

    # 提取 source=rag 字段
    rag_fields = [
        {"name": f.get("name"), "description": f.get("description", "")}
        for f in template.get("fields", [])
        if f.get("source") == "rag"
    ]

    # 全局 RAG 摘要
    global_summary = "（无全局 RAG 结果）"
    if rag_context:
        global_summary = json.dumps(
            [{"query": d["query"], "score": d.get("score", 0)} for d in rag_context[:10]],
            ensure_ascii=False, indent=2,
        )

    # 构建 Plan 摘要
    plan_descriptions = [
        {
            "index": i,
            "id": p.get("id", f"plan_{i}"),
            "name": p.get("name", ""),
            "description": p.get("description", ""),
            "type": p.get("type", ""),
        }
        for i, p in enumerate(plans)
    ]

    # Step 1: LLM 判断每个 Plan 是否需要独立 RAG
    prompt = _get_plan_rag_planner_prompt().format(
        offer_overview=offer_overview,
        global_rag_summary=global_summary,
        plan_list=json.dumps(plan_descriptions, ensure_ascii=False, indent=2),
        plan_count=len(plan_descriptions),
        template_fields=json.dumps(rag_fields, ensure_ascii=False, indent=2),
    )

    try:
        response = llm.invoke([
            SystemMessage(content="你只返回 JSON，不要任何额外文本。"),
            HumanMessage(content=prompt),
        ])
        result = _parse_json_response(response.content)
        plan_rag_map = result.get("plan_rag_map", {})
    except Exception as e:
        logger.warning(f"Plan RAG 规划失败: {e}，所有 Plan 使用全局 RAG")
        return {"plan_rag_contexts": {}}

    # Step 2: 为需要 RAG 的 Plan 执行检索
    plan_rag_contexts: dict[int, list[dict[str, Any]]] = {}
    needs_rag_count = 0

    if not rag_cfg.get("base_url"):
        logger.info("plan_rag_planner: RAG API 不可用，跳过 Plan 级检索")
        return {"plan_rag_contexts": {}}

    try:
        min_score = rag_cfg.get("min_score", 0.3)
        config = RAGAPIConfig(
            base_url=rag_cfg["base_url"],
            api_key=rag_cfg.get("api_key"),
            endpoints=rag_cfg.get("endpoints", []),
            timeout=rag_cfg.get("timeout", 30),
            max_retries=rag_cfg.get("max_retries", 2),
            default_top_k=rag_cfg.get("default_top_k", 5),
            min_score=min_score,
        )
        client = RAGClient(config)

        for plan_index_str, plan_info in plan_rag_map.items():
            plan_index = int(plan_index_str)
            if plan_index >= len(plans):
                continue

            if not plan_info.get("need_rag", False):
                continue

            queries = plan_info.get("queries", [])
            if not queries:
                continue

            plan_name = plan_info.get("plan_name", f"plan_{plan_index}")
            logger.info(f"Plan[{plan_index}] '{plan_name}' → {len(queries)} 条专属查询")

            try:
                results = client.batch_search(queries)
                plan_docs: list[dict[str, Any]] = []
                for query, docs in results.items():
                    for doc in docs:
                        score = doc.get("score", 0)
                        if score < min_score:
                            continue
                        plan_docs.append({
                            "query": query,
                            "content": doc.get("content", ""),
                            "score": score,
                            "metadata": doc.get("metadata", {}),
                        })
                plan_docs.sort(key=lambda x: x["score"], reverse=True)
                plan_rag_contexts[plan_index] = plan_docs[:8]
                needs_rag_count += 1
                logger.info(
                    f"  Plan[{plan_index}] 检索完成: {len(plan_docs[:8])} 条文档"
                )
            except Exception as e:
                logger.warning(f"  Plan[{plan_index}] 检索失败: {e}")

        client.close()
    except Exception as e:
        logger.error(f"Plan RAG 检索阶段异常: {e}")

    logger.info(
        f"Plan RAG 规划完成: {needs_rag_count}/{len(plans)} 个 Plan 需要专属检索"
    )
    return {"plan_rag_contexts": plan_rag_contexts}


# ═════════════════════════════════════════════════════════════════
#  Phase 4: 并行 Plan 处理 (LangGraph Send)
# ═════════════════════════════════════════════════════════════════

_PLAN_WORKER_PROMPT: str | None = None


def _get_plan_worker_prompt() -> str:
    global _PLAN_WORKER_PROMPT
    if _PLAN_WORKER_PROMPT is None:
        _PLAN_WORKER_PROMPT = load_prompt("plan_worker")
    return _PLAN_WORKER_PROMPT


def plan_worker(state: AgentState, llm: ChatOpenAI) -> dict[str, Any]:
    """
    单个 plan 的画像片段生成 (LLM × 1)

    输入：offer 总览 + RAG 上下文 + 该 plan 的完整 JSON
    输出：该 plan 的画像片段 dict，追加到 plan_results
    """
    plan_index = state["plan_index"]
    plans = state.get("plans", [])
    offer_overview = state.get("offer_overview", "")
    rag_context = state.get("rag_context", [])
    plan_rag_contexts = state.get("plan_rag_contexts", {})

    if plan_index >= len(plans):
        logger.warning(f"plan_index={plan_index} 超出范围 (total={len(plans)})")
        return {"plan_results": []}

    plan_data = plans[plan_index]
    plan_id = plan_data.get("id", f"plan_{plan_index}")
    plan_name = plan_data.get("name", "Unknown")
    logger.info(f"[worker-{plan_index}] 开始处理: {plan_id} ({plan_name})")

    # 合并全局 RAG + Plan 专属 RAG
    merged_context = list(rag_context)
    plan_docs = plan_rag_contexts.get(plan_index, [])
    if plan_docs:
        merged_context = merged_context + plan_docs
        logger.info(
            f"[worker-{plan_index}] RAG: 全局{len(rag_context)}条 "
            f"+ 专属{len(plan_docs)}条 = {len(merged_context)}条"
        )

    # 构建 RAG 文本
    rag_text = "（无 RAG 检索结果）"
    if merged_context:
        items = []
        for j, ctx in enumerate(merged_context, 1):
            items.append(f"[{j}] {ctx['query']}\n    {ctx['content'][:300]}")
        rag_text = "\n".join(items)

    prompt = _get_plan_worker_prompt().format(
        offer_overview=offer_overview,
        plan_index=plan_index,
        plan_json=json.dumps(plan_data, ensure_ascii=False, indent=2),
        rag_context=rag_text,
    )

    try:
        response = llm.invoke([
            SystemMessage(content="你只返回 JSON，不要任何额外文本。"),
            HumanMessage(content=prompt),
        ])
        plan_profile = _parse_json_response(response.content)

        # 注入 plan 标识
        plan_profile["_plan_id"] = plan_id
        plan_profile["_plan_name"] = plan_name
        plan_profile["_plan_index"] = plan_index

        logger.info(f"[worker-{plan_index}] 完成: {plan_id}")
        return {"plan_results": [plan_profile]}
    except Exception as e:
        logger.error(f"[worker-{plan_index}] 处理失败: {e}")
        return {
            "plan_results": [{
                "_plan_id": plan_id,
                "_plan_name": plan_name,
                "_plan_index": plan_index,
                "_error": str(e),
            }],
        }


# ═════════════════════════════════════════════════════════════════
#  Phase 5: 汇总生成 + 校验 (LLM × 1)
# ═════════════════════════════════════════════════════════════════

_FINAL_PROMPT: str | None = None


def _get_final_prompt() -> str:
    global _FINAL_PROMPT
    if _FINAL_PROMPT is None:
        _FINAL_PROMPT = load_prompt("generate_final_profile")
    return _FINAL_PROMPT


def generate_final_profile(state: AgentState, llm: ChatOpenAI) -> dict[str, Any]:
    """
    汇总所有并行 worker 结果 + 总览 + RAG → 生成最终 offer 画像
    """
    offer_overview = state.get("offer_overview", "")
    plan_results = state.get("plan_results", [])
    rag_context = state.get("rag_context", [])
    template = state.get("profile_template", {})
    basic_info = state.get("tmf_basic_info", {})

    plan_count = len(plan_results)

    # 如果没有 plan_results（无 plan），回退到直接基于基本信息生成
    if plan_count == 0:
        logger.info("无 plan_results，基于基本信息 + TMF 直接生成")
        return _generate_simple_profile(state, llm)

    # 按 plan_index 排序
    plan_results_sorted = sorted(plan_results, key=lambda x: x.get("_plan_index", 0))

    # 构建 RAG 文本
    rag_text = "（无 RAG 检索结果）"
    if rag_context:
        items = []
        for j, ctx in enumerate(rag_context, 1):
            items.append(f"[{j}] {ctx['query']}\n    {ctx['content'][:300]}")
        rag_text = "\n".join(items)

    # 模板字段描述
    fields_desc = json.dumps(
        [{"name": f["name"], "description": f.get("description", ""),
          "source": f.get("source", ""), "type": f.get("type", "string")}
         for f in template.get("fields", [])],
        ensure_ascii=False, indent=2,
    )

    prompt = _get_final_prompt().format(
        offer_basic=json.dumps(basic_info, ensure_ascii=False, indent=2),
        offer_overview=offer_overview,
        plan_count=plan_count,
        plan_results=json.dumps(plan_results_sorted, ensure_ascii=False, indent=2),
        rag_context=rag_text,
        template_fields=fields_desc,
    )

    try:
        response = llm.invoke([
            SystemMessage(content="你只返回 JSON，严格按字段定义结构，不要任何额外文本。"),
            HumanMessage(content=prompt),
        ])
        profile = _parse_json_response(response.content)
        logger.info(f"最终画像生成完成: {len(profile)} 个顶层字段, 来自 {plan_count} 个 plan")
        return {"profile_output": profile}
    except Exception as e:
        logger.error(f"最终画像生成失败: {e}")
        return {
            "profile_output": {"error": str(e), "plan_count": plan_count},
            "messages": [AIMessage(content=f"[FinalProfile] 生成失败: {e}")],
        }


def _generate_simple_profile(state: AgentState, llm: ChatOpenAI) -> dict[str, Any]:
    """无 plan 时的简化画像生成（兼容旧数据）"""
    template = state.get("profile_template", {})
    tmf_input = state.get("tmf_input", {})
    rag_context = state.get("rag_context", [])
    offer_overview = state.get("offer_overview", "")

    _GEN_SIMPLE_PROMPT: str | None = None
    try:
        _GEN_SIMPLE_PROMPT = load_prompt("generate_profile")
    except FileNotFoundError:
        _GEN_SIMPLE_PROMPT = load_prompt("generate_final_profile")

    rag_text = "（无 RAG 检索结果）"
    if rag_context:
        items = []
        for j, ctx in enumerate(rag_context, 1):
            items.append(f"[{j}] {ctx['query']}\n    {ctx['content'][:500]}")
        rag_text = "\n".join(items)

    prompt = _GEN_SIMPLE_PROMPT.format(
        template_json=json.dumps({"fields": template.get("fields", [])}, ensure_ascii=False, indent=2),
        tmf_json=json.dumps(tmf_input, ensure_ascii=False, indent=2),
        rag_context=rag_text,
        offer_overview=offer_overview or tmf_input.get("description", ""),
        plan_results="（无 Plan 数据）",
    )

    try:
        response = llm.invoke([
            SystemMessage(content="你只返回 JSON，严格按字段定义结构，不要额外文本。"),
            HumanMessage(content=prompt),
        ])
        return {"profile_output": _parse_json_response(response.content)}
    except Exception as e:
        return {"profile_output": {"error": str(e)}}


def validate_profile(state: AgentState) -> dict[str, Any]:
    """
    校验最终生成的画像 JSON

    检查 required 字段完整性、类型匹配。
    即使校验失败也保留 final_output，由调用方决定是否采纳。
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
        expected_type = field.get("type", "string")
        value = profile.get(name)

        if required:
            if value is None:
                errors.append(f"缺少必填字段: {name}")
                continue
            if isinstance(value, (list, str, dict)) and len(value) == 0:
                errors.append(f"必填字段为空: {name}")

        if value is not None and not _check_type(value, expected_type):
            errors.append(
                f"字段类型错误: {name} 期望 {expected_type}, "
                f"实际 {type(value).__name__}"
            )

    if errors:
        for err in errors:
            logger.warning(f"  - {err}")
        return {"validation_errors": errors, "final_output": profile}

    logger.info("画像校验通过")
    return {"validation_errors": [], "final_output": profile}


def _check_type(value: Any, expected: str) -> bool:
    type_map: dict[str, Any] = {
        "string": str,
        "number": (int, float),
        "integer": int,
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    expected_types = type_map.get(expected)
    return expected_types is None or isinstance(value, expected_types)


# ═════════════════════════════════════════════════════════════════
#  工具函数
# ═════════════════════════════════════════════════════════════════

def _parse_json_response(raw: str | list | dict) -> dict[str, Any]:
    """解析 LLM 输出的 JSON（自动剥离 markdown 代码块）"""
    if not isinstance(raw, str):
        raw = str(raw)
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1])
        else:
            text = "\n".join(lines[1:])
    return json.loads(text)
