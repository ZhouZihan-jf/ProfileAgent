# ProfileAgent — 运营商 Offer 画像生成 Agent

基于 **LangGraph** 构建的多 Plan 并行架构 Agent，将 **TMF (TM Forum)** 格式的运营商套餐 JSON 自动转换为结构化 **Offer 画像 JSON**。

---

## 目录

- [项目背景](#项目背景)
- [核心设计](#核心设计)
- [架构概览](#架构概览)
- [工作流详解](#工作流详解)
- [状态设计](#状态设计)
- [模块设计](#模块设计)
- [目录结构](#目录结构)
- [快速开始](#快速开始)
- [配置说明](#配置说明)

---

## 项目背景

运营商的产品套餐（Offer）通常以 TMF 格式描述，结构庞大且嵌套深——一个 Offer 可能包含数十个 Plan，每个 Plan 内部又嵌套 Policy → Pattern → Action 等多层结构。直接让 LLM 处理完整的 TMF JSON 面临两个核心挑战：

1. **Token 上限**：单个 Offer 的 TMF JSON 可达数千字符，加上 prompt、模板、RAG 上下文后极易超出窗口
2. **Plan 独立性**：每个 Plan 的画像分析彼此独立，串行处理效率低

本项目的方案是：**先瘦身 → 总览生成 → 并行逐 Plan 处理 → 汇总**，利用 LangGraph 的 `Send` 原语实现 Map-Reduce 并行模式，同时引入按需 RAG 检索来补充外部行业知识。

---

## 核心设计

### 设计原则

| 原则 | 实现方式 |
|------|---------|
| **减少 Token 消耗** | JSON 瘦身（递归裁剪非关键字段），总览阶段只传 Plan 摘要，worker 阶段只传单个 Plan 完整数据 |
| **并行加速** | LangGraph `Send` API 实现 fan-out → [worker × N] → fan-in |
| **按需 RAG** | 由 LLM 自主判断是否需要知识检索，自动生成检索查询 |
| **全链路降级** | 配置文件缺失、LLM 调用失败、RAG 不可用时均有 fallback |
| **配置驱动** | 画像模板、RAG 规则、Prompt 模板全部外部化 |

### 关键决策

**Q: 为什么先在总览阶段只传 Plan 摘要，而不是完整 JSON？**

总览阶段的目标是理解 Offer 全局含义并决定是否需要 RAG。Plan 摘要（id + name + description）足以支撑这个判断。把完整的 Plan JSON 放在 worker 阶段单独处理，避免总览占用过多上下文。

**Q: 为什么让 LLM 决定是否需要 RAG，而不是用规则？**

RAG 调用有成本（延迟 + 资源）。让 LLM 基于 template 的 `source` 字段和 TMF 内容判断，比纯规则更灵活——LLM 可以判断"虽然配置了 rag 字段，但 TMF 内容已足够"。

**Q: 为什么用 `Send` 而不是 `SubGraph`？**

`Send` 是 LangGraph 的 Map-Reduce 原语，天然支持并行 + reducer 自动合并。Plan worker 之间无依赖，适合 fan-out/fan-in 模式。`SubGraph` 更适用于有依赖关系的子流程。

---

## 架构概览

```
                    ┌──────────────────────────────┐
                    │    CLI 入口 (src/main.py)      │
                    │  python -m src.main --input   │
                    │  data/sample_tmf.json         │
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────▼───────────────┐
                    │      compile_agent(llm)       │
                    │   LangGraph StateGraph v2     │
                    │   + MemorySaver Checkpointer  │
                    └──────────────┬───────────────┘
                                   │
     ┌─────────────────────────────┼─────────────────────────────┐
     │                             ▼                             │
     │  Phase 1 ─── parse_tmf → slim_json → load_config         │
     │    (纯代码)    校验 +        JSON     加载模板/规则/       │
     │              类型转换      瘦身      RAG API 配置          │
     │                             │                             │
     │              Phase 2 ─── generate_overview (LLM × 1)      │
     │                    ┌────────┴────────┐                    │
     │              need_rag?               no                   │
     │              │ yes                    │                    │
     │   Phase 3 ─── rag_query              │                    │
     │              │                        │                    │
     │              └────────┬───────────────┘                    │
     │                       ▼                                    │
     │              dispatch_plans (passthrough)                  │
     │                       │                                    │
     │     ┌─────────────────┼─────────────────┐                 │
     │     │                 │                 │                 │
     │  Send(plan_worker,  Send(plan_worker,  Send(plan_worker,  │
     │    {plan_index:0})   {plan_index:1})   {plan_index:2})    │
     │     │                 │                 │                 │
     │     ▼                 ▼                 ▼                 │
     │  plan_worker      plan_worker       plan_worker           │
     │   (LLM × 1)        (LLM × 1)         (LLM × 1)            │
     │     │                 │                 │                 │
     │     └─────────────────┼─────────────────┘                 │
     │                       │                                   │
     │          Reducer 自动合并 plan_results                     │
     │                       │                                   │
     │  Phase 5 ─── generate_final_profile → validate_profile    │
     │               (LLM × 1)                     │             │
     │                                             │             │
     │                                       output/*.json       │
     └───────────────────────────────────────────────────────────┘
```

**LLM 调用次数**：1（总览）+ N（Plan worker）+ 1（汇总）= **N + 2 次**，其中 N 个 worker 并行执行。

---

## 工作流详解

### Phase 1：预处理（纯代码，无 LLM 调用）

#### `parse_tmf` — TMF 校验

- 字符串 → dict 反序列化
- 调用 `validate_tmf_input()` 检查必填字段（`id`、`name`）
- 确保 `plan_list` 字段存在（缺失时降级为空列表）

#### `slim_json` — JSON 瘦身

递归裁剪 TMF 结构，减少后续阶段的 Token 消耗：

| 层级 | 保留字段 |
|------|---------|
| 顶层 | `id`, `name`, `description`, `category`, `characteristics`, `product_specification`, `bundled_product_offering` |
| Plan | `id`, `name`, `description`, `type`, `status`, `lifecycle_status` |
| Policy | `id`, `name`, `description`, `type`, `priority`, `status` |
| Pattern | `id`, `name`, `description`, `type` |
| Action | `id`, `name`, `description`, `type`, `value`, `unit` |

瘦身结果分区存储：
- `tmf_basic_info`：顶层信息（不含 plan_list）
- `plans`：瘦身后的 Plan 列表

#### `load_config` — 配置加载

从 `config/` 目录加载 3 类配置：

| 配置项 | 文件 | 加载到 |
|--------|------|--------|
| 画像模板 | `templates/default_profile.json` | `profile_template` |
| RAG 规则 | `rules/rag_rules.md` | `rules_md` |
| RAG API | `rag_config.yaml` | `rag_api_config` |

所有配置缺失时均有 fallback 默认值，不会阻断流程。

---

### Phase 2：总览生成（LLM × 1）

`generate_overview(state, llm)` 负责：

1. **构建 Plan 摘要**：从每个 Plan 仅提取 `index`、`id`、`name`、`description`，不传入完整 JSON
2. **提取 RAG 需求字段**：遍历模板，找出 `source == "rag"` 的字段
3. **组装 Prompt**：注入 rules、offer_basic、plan_summaries、template_fields
4. **调用 LLM**，要求返回：

```json
{
  "overview": "套餐总览文本...",
  "need_rag": true,
  "reasoning": "需要 RAG 的原因...",
  "queries": ["5G套餐市场趋势", "竞争对手相似产品"]
}
```

5. **降级**：LLM 调用失败时，以 `basic_info.description` 作为 overview，`need_rag = false`

**条件路由**：`need_rag == true AND queries 非空` → `rag_query`；否则直接 → `dispatch_plans`

---

### Phase 3：RAG 检索（可选）

`rag_query(state)` 负责：

1. 检查 `need_rag` 标记和 `queries` 列表
2. 从 `rag_api_config` 创建 `RAGClient`（基于 `httpx`）
3. 调用 `batch_search(queries)`，每条查询取 top-3 文档
4. 返回 `rag_context: list[dict]`，每个 doc 包含 `{query, content, score, metadata}`

**容错机制**：
- `need_rag = false` 或无 queries → 跳过
- RAG API 未配置 → 跳过
- 网络/解析异常 → 返回空 context + 错误消息

---

### Phase 4：并行 Plan 处理（LLM × N）

这是 v2 的核心改进——使用 LangGraph `Send` 实现 Map-Reduce 并行。

#### 分发节点 `dispatch_plans`

- 本身是一个 `_passthrough` 节点（不改变状态）
- 完成后由 `_fan_out_plans()` 路由：

```python
plans = state.get("plans", [])
if not plans:
    return "generate_final_profile"  # 无 Plan，跳过并行

return [
    Send("plan_worker", {"plan_index": i})
    for i in range(len(plans))
]
```

- 有 Plan → 每个 Plan 生成一个 `Send("plan_worker", {"plan_index": i})`，LangGraph 并行调度
- 无 Plan → 直接跳到 Phase 5

#### Worker 节点 `plan_worker(state, llm)`

每个 worker 实例接收一个独立的 `plan_index`：

1. 从 `state["plans"]` 取值 `plans[plan_index]`
2. 组装 prompt：offer_overview + plan JSON + RAG context
3. 调用 LLM 生成该 Plan 的**画像片段**（JSON dict）
4. 注入标识：`_plan_id`、`_plan_name`、`_plan_index`
5. 返回 `{"plan_results": [plan_profile]}`

#### Reducer 自动合并

`plan_results` 字段在 `state.py` 中配置了自定义 reducer：

```python
def _merge_plan_results(existing, new):
    if existing is None:
        existing = []
    if new is None:
        return existing
    return existing + new
```

当所有 `Send` 实例完成后，LangGraph 自动合并所有 worker 的 `plan_results`。

---

### Phase 5：汇总生成 + 校验（LLM × 1）

#### `generate_final_profile(state, llm)`

- **有 Plan**：按 `_plan_index` 排序 `plan_results`，融合 `offer_overview` + `rag_context` → 调用 LLM → 最终画像 JSON
- **无 Plan**（降级路径）：直接基于 `tmf_input` 调用简化 prompt

#### `validate_profile(state)`

校验最终画像的完整性和正确性：

- 逐一检查模板中 `required: true` 的字段是否存在且非空
- 类型匹配检查：`string` / `number` / `integer` / `boolean` / `array` / `object`
- **无论校验是否通过**，都将画像写入 `final_output`（由调用方决定是否采纳）

---

## 状态设计

`AgentState` 定义为 `TypedDict`，按阶段划分为 5 组：

```
Phase 1 → Phase 2 → Phase 3 → Phase 4 (并行) → Phase 5
────────────────────────────────────────────────────────
tmf_input         offer_overview    rag_context    plan_index        profile_output
tmf_basic_info    need_rag                         plan_results*     validation_errors
plans             rag_queries                       (reducer)        final_output
rules_md
profile_template
rag_api_config
messages  (LangChain BaseMessage 序列，贯穿全局)
```

`* plan_results` 使用 `_merge_plan_results` reducer 实现 Send 并行分支收敛时自动合并。

**状态字段的分类**：

| 来源 | 字段 |
|------|------|
| 用户输入 | `tmf_input` |
| Phase 1 纯代码输出 | `tmf_basic_info`, `plans`, `rules_md`, `profile_template`, `rag_api_config` |
| Phase 2 LLM 输出 | `offer_overview`, `need_rag`, `rag_queries` |
| Phase 3 RAG 输出 | `rag_context` |
| Phase 4 路由注入 | `plan_index` |
| Phase 4 Worker 输出 | `plan_results` |
| Phase 5 输出 | `profile_output`, `validation_errors`, `final_output` |

---

## 模块设计

### `src/agent/` — Agent 核心

| 文件 | 职责 |
|------|------|
| `state.py` | `AgentState` TypedDict 定义 + `_merge_plan_results` reducer |
| `graph.py` | LangGraph StateGraph 组装、LLM 闭包注入、条件路由、Send fan-out |
| `nodes.py` | 全部 7 个节点的实现 + `_parse_json_response` 工具函数 |

**模块间依赖**：

```
graph.py ──imports──→ nodes.py ──imports──→ config.py
   │                      │                    │
   └──imports──→ state.py └──imports──→ rag/client.py
                                          │
                                     config.py (RAGAPIConfig)
```

### `src/config/` — 配置层

| 文件 | 职责 |
|------|------|
| `config.py` | Pydantic 数据模型 + 文件加载器 + TMF 校验 |

**Pydantic 模型**：

```python
ProfileField       # 画像字段定义（name/type/description/required/default/source）
ProfileTemplate    # 画像模板（name/version/fields）
RAGEndpoint        # RAG API 端点（name/path/method/headers）
RAGAPIConfig       # RAG API 全局配置
TMFOffer           # TMF 简化校验模型
```

**文件加载器**均通过 `_get_config_dir()` 定位项目 `config/` 目录：

```python
load_profile_template(name)  # config/templates/{name}_profile.json
load_rag_config()             # config/rag_config.yaml
load_rules()                  # config/rules/rag_rules.md
load_prompt(name)             # config/prompts/{name}.md
validate_tmf_input(data)      # 校验 id/name 必填
```

### `src/rag/` — RAG 客户端

基于 `httpx.Client` 的 RESTful API 封装：

```
RAGClient
├── __init__(config: RAGAPIConfig)   # 初始化 HTTP 客户端
├── search(query, top_k, endpoint)   # 单次搜索
├── batch_search(queries)            # 批量搜索
├── close()                          # 手动关闭连接
└── _parse_response(data)            # 兼容 3 种 API 响应格式
```

**响应格式兼容**：
1. `{"results": [{content, score, metadata}]}`
2. `{"documents": [{text, similarity}]}`
3. `{"data": [...]}`

**容错**：支持自动重试（`max_retries`）、超时、Bearer Token 认证。

### `src/main.py` — CLI 入口

```
python -m src.main --input data/sample_tmf.json [--verbose]
```

流程：
1. 加载 `.env` → 创建 `ChatOpenAI`（兼容 DeepSeek/OpenAI）
2. 读取 TMF JSON → `_make_initial_state()`
3. `compile_agent(llm)` → `agent.invoke()`
4. 输出写入 `output/{input_name}_profile.json`

---

## 目录结构

```
ProfileAgent/
├── src/
│   ├── main.py                      # CLI 入口 + Agent 运行
│   ├── agent/
│   │   ├── state.py                 # AgentState TypedDict 定义
│   │   ├── graph.py                 # LangGraph StateGraph 组装
│   │   └── nodes.py                 # 7 个工作流节点实现
│   ├── config/
│   │   └── config.py                # Pydantic 模型 + 配置加载
│   └── rag/
│       └── client.py                # RAG RESTful API 客户端
├── config/
│   ├── rag_config.yaml              # RAG API 连接配置
│   ├── prompts/
│   │   ├── generate_overview.md     # Phase 2: 总览生成 prompt
│   │   ├── plan_worker.md           # Phase 4: Plan 分析 prompt
│   │   ├── generate_final_profile.md# Phase 5: 最终汇总 prompt
│   │   ├── generate_profile.md      # 降级/无 Plan 时 prompt
│   │   └── rag_decision.md          # RAG 决策 prompt（备用）
│   ├── rules/
│   │   └── rag_rules.md             # RAG 调用规则
│   └── templates/
│       └── default_profile.json     # 画像输出模板（12 字段）
├── data/
│   └── sample_tmf.json              # 示例 TMF 输入
├── output/                          # 生成的画像输出目录
├── requirements.txt                 # Python 依赖
├── pyrightconfig.json               # Pyright 类型检查配置
├── .env.example                     # 环境变量模板
└── README.md                        # 本文档
```

---

## 快速开始

### 1. 环境准备

```powershell
# 创建虚拟环境
python -m venv .venv
.venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置 API Key

```powershell
copy .env.example .env
# 编辑 .env，填入 DeepSeek API Key
```

`.env` 内容：

```ini
DEEPSEEK_API_KEY=sk-your-key
OPENAI_API_KEY=sk-your-key
OPENAI_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat
```

### 3. 运行

```powershell
python -m src.main --input data/sample_tmf.json --verbose
```

输出写入 `output/sample_tmf_profile.json`。

### 4. 自定义输入

- 准备 TMF 格式的 JSON 文件，放入 `data/` 目录
- 确保 JSON 顶层包含 `id`、`name` 和 `plan_list` 字段

---

## 配置说明

### 画像模板 (`config/templates/default_profile.json`)

定义最终输出 JSON 的字段结构，每个字段配置：

```json
{
  "name": "competitive_advantages",
  "type": "array",
  "required": false,
  "description": "竞争优势",
  "source": "rag",
  "prompt_instruction": "提取本套餐相对竞品的核心优势"
}
```

- `source`：`tmf`（从 TMF 提取）/ `model`（LLM 生成）/ `rag`（RAG 检索）/ `rule`（规则计算）
- `required`：`true` 时 validate_profile 会检查字段存在性

### RAG 配置 (`config/rag_config.yaml`)

```yaml
base_url: "http://localhost:8000/api/v1"
api_key: ""
timeout: 30
max_retries: 2
default_top_k: 5
endpoints:
  - name: search
    path: /rag/search
    method: POST
```

RAG 客户端默认使用第一个 `endpoint`，可通过 `search()` 的 `endpoint_name` 参数切换。

### Prompt 模板 (`config/prompts/*.md`)

所有 prompt 使用 Python `str.format()` 风格的占位符，在运行时由节点注入实际数据：

| Prompt | 占位符 |
|--------|--------|
| `generate_overview.md` | `{rules}`, `{offer_basic}`, `{plan_summaries}`, `{plan_count}`, `{template_fields}`, `{rag_field_count}` |
| `plan_worker.md` | `{offer_overview}`, `{plan_index}`, `{plan_json}`, `{rag_context}` |
| `generate_final_profile.md` | `{template_fields}`, `{offer_basic}`, `{offer_overview}`, `{plan_count}`, `{plan_results}`, `{rag_context}` |

---

## 技术栈

| 组件 | 版本 | 用途 |
|------|------|------|
| **LangGraph** | ≥ 0.2.0 | 状态图编排 + Send 并行 + Checkpointer |
| **LangChain** | ≥ 0.3.0 | LLM 调用封装（ChatOpenAI） |
| **Pydantic** | ≥ 2.0.0 | 配置/输入数据模型校验 |
| **httpx** | ≥ 0.27.0 | RAG API HTTP 客户端 |
| **PyYAML** | ≥ 6.0 | RAG 配置文件解析 |
| **python-dotenv** | ≥ 1.0.0 | 环境变量加载 |
| **DeepSeek API** | — | 默认 LLM 后端（兼容 OpenAI SDK） |
