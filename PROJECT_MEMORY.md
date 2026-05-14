# ProfileAgent 项目记忆文档

> 生成时间: 2025-05-14
> 最后更新: 2025-05-14
> 项目路径: `f:/projects/ProfileAgent`

> **架构决策**: prompt 模板全部存放于 `config/prompts/`，通过 `load_prompt()` 加载，支持热编辑。
> **架构精简**: 所有 Agent 节点合并为 `nodes.py`，配置模型+加载器合并为 `config.py`，源码从17个文件精简至9个。

---

## 一、项目概述

**ProfileAgent** 是一个基于 LangGraph 的 AI Agent，核心功能是：**输入运营商的 TMF 格式套餐 Offer JSON，自动生成对应的 Offer 画像 JSON**。

### 核心设计理念

| 特性 | 实现 |
|------|------|
| **可配置画像模板** | 通过 YAML 定义输出 JSON 结构，无需改代码即可切换不同画像 schema |
| **模型自主 RAG 决策** | 模型根据 `rules.md` 规则自行判断是否调用 RAG API，而非硬编码 |
| **多源字段支持** | 每个画像字段标记 `source`：`tmf`(提取) / `model`(LLM生成) / `rag`(检索) / `rule`(规则计算) |
| **RAG API 通过 RESTful 接入** | `RAGClient` 类统一管理 API 调用，支持多种响应格式自动适配 |
| **条件路由** | LangGraph 条件边实现按需跳过 RAG 节点 |

---

## 二、技术栈

```
Python 3.10+
├── langgraph >= 0.2.0       # Agent 工作流框架
├── langchain >= 0.3.0       # LLM 链式调用
├── langchain-openai >= 0.2.0 # OpenAI 兼容 LLM
├── pydantic >= 2.0.0        # 配置模型校验
├── pyyaml >= 6.0            # YAML 配置解析
├── httpx >= 0.27.0          # RAG API HTTP 客户端
└── python-dotenv >= 1.0.0   # 环境变量管理
```

完整依赖见 `requirements.txt`。

### 运行环境

- **Python**: 3.12 (uv 虚拟环境 `.venv/`)
- **依赖安装**: `cd [project] && uv pip install -r requirements.txt`
- **类型检查**: basedpyright，自定义配置 `pyrightconfig.json`
  - 基础 strict 模式启用
  - `reportAny` / `reportExplicitAny`: 禁用（项目大量字典操作，此类规则无益）

---

## 三、项目结构

```
ProfileAgent/
├── PROJECT_MEMORY.md                  # 本文档 - 项目记忆
├── .env.example                       # 环境变量模板
├── .gitignore
├── requirements.txt
│
├── config/                            # 可配置项（全外置，无需改代码）
│   ├── prompts/
│   │   ├── generate_profile.md        # 画像生成 prompt（含 {template_json} 等占位符）
│   │   └── rag_decision.md            # RAG 决策 prompt（含 {rules} {tmf_summary} 等占位符）
│   ├── templates/
│   │   └── default_profile.json       # 画像模板 - 定义输出 JSON 结构
│   ├── rules/
│   │   └── rag_rules.md               # RAG 调用规则 - 模型决策依据
│   └── rag_config.yaml                # RAG API 连接信息
│
├── data/
│   └── sample_tmf.json                # 示例 TMF 输入（5G畅享Pro套餐）
│
└── src/
    ├── __init__.py
    ├── main.py                        # 入口: CLI 解析 + Agent 运行
    │
    ├── agent/
    │   ├── __init__.py
    │   ├── state.py                   # LangGraph AgentState 类型定义
    │   ├── graph.py                   # 状态图组装 + 条件路由
    │   └── nodes.py                   # 6个节点函数（parse_tmf → validate_profile）
    │
    ├── config/
    │   ├── __init__.py
    │   └── config.py                  # Pydantic 模型 + 文件加载器（合并 models+loader）
    │
    └── rag/
        ├── __init__.py
        └── client.py                  # RAG RESTful API 客户端
```

---

## 四、LangGraph 工作流

### 流程图

```
                    ┌──────────────┐
                    │  parse_tmf   │ ← 入口：校验 TMF JSON
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │ load_config  │ ← 加载模板、规则、RAG 配置
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │ rag_decision │ ← ★ 模型自主决策
                    └──┬──────┬────┘
                       │      │
              need_rag=true  need_rag=false
                       │      │
               ┌───────▼──┐   │
               │ rag_query│   │
               └──────┬───┘   │
                      │       │
               ┌──────▼───────▼──┐
               │generate_profile│ ← LLM 融合上下文生成画像
               └──────┬─────────┘
                      │
               ┌──────▼──────────┐
               │validate_profile │ ← 校验后输出最终 JSON
               └──────┬──────────┘
                      │
                   ┌──▼──┐
                   │ END │
                   └─────┘
```

### 节点职责

| # | 节点 | 位置 | 类型 | 职责 |
|---|------|------|------|------|
| ① | `parse_tmf` | `nodes.py` | 纯函数 | 校验 TMF 输入必填字段 (id, name)，支持 str→dict 转换 |
| ② | `load_config` | `nodes.py` | 纯函数 | 从 `config/` 加载模板 JSON、规则 MD、RAG API 配置，有 fallback |
| ③ | `rag_decision` | `nodes.py` | LLM调用 | 模型读取 rules.md + 模板字段 + TMF 数据 → 输出 `need_rag` + `queries` JSON |
| ④ | `rag_query` | `nodes.py` | 纯函数 | 通过 RAGClient 批量检索，结果存入 `rag_context` |
| ⑤ | `generate_profile` | `nodes.py` | LLM调用 | 融合 TMF + RAG + 模板 → LLM 生成结构化画像 JSON |
| ⑥ | `validate_profile` | `nodes.py` | 纯函数 | 校验 required 字段、类型匹配，输出 `final_output` |

### AgentState 结构

```python
class AgentState(TypedDict):
    # 输入
    tmf_input: dict[str, Any]           # 原始 TMF JSON
    # 配置
    rules_md: str                       # RAG 使用规则 Markdown
    profile_template: dict[str, Any]    # 画像模板 (字段定义)
    rag_api_config: dict[str, Any]      # RAG API 连接信息
    # 消息
    messages: Annotated[Sequence[BaseMessage], add_messages]
    # RAG 相关
    need_rag: bool                      # 是否调用 RAG
    rag_queries: list[str]              # 模型生成的查询列表
    rag_context: list[dict]             # RAG 返回结果
    # 输出
    profile_output: Optional[dict]      # LLM 生成的画像
    validation_errors: list[str]        # 校验错误
    final_output: Optional[dict]        # 最终输出
```

---

## 五、核心模块详解

### 5.1 配置系统 (`src/config/config.py`)

一个文件包含两部分：

**数据模型** — Pydantic 类：
- `ProfileField`: 单个画像字段定义（name, type, source, required, prompt_instruction...）
- `ProfileTemplate`: 画像模板整体结构
- `RAGEndpoint`: RAG API 端点（name, path, method, headers）
- `RAGAPIConfig`: RAG API 完整配置（base_url, api_key, endpoints, timeout...）
- `TMFOffer`: TMF 输入模型

**加载器函数**：
- `load_prompt(name)` — 加载 `config/prompts/{name}.md`，支持惰性缓存
- `load_profile_template(name)` — 加载 `config/templates/{name}_profile.json`
- `load_rag_config()` — 加载 `config/rag_config.yaml`
- `load_rules()` — 加载 `config/rules/rag_rules.md`
- `validate_tmf_input()` — 校验 TMF JSON 必填字段

### 5.2 RAG 客户端 (`src/rag/client.py`)

```python
class RAGClient:
    def __init__(self, config: RAGAPIConfig)
    def search(query, top_k, endpoint_name, **extra_params) -> list[dict]
    def batch_search(queries, top_k) -> dict[str, list[dict]]
    def get_endpoint(name) -> RAGEndpoint
```

关键特性：
- **多格式响应解析**：自动适配 3 种常见 API 响应格式 (`results`/`documents`/`data`)
- **重试机制**：超时自动重试（可配置次数）
- **端点选择**：通过 `get_endpoint(name)` 切换不同 API 端点
- **批量检索**：`batch_search` 逐条查询并聚合结果

### 5.3 rag_decision 节点 (`src/agent/nodes.py` 中的 `rag_decision` 函数)

**这是整个 Agent 最核心的决策节点**：

1. 提取模板中 `source=rag` 的字段
2. 若无 RAG 字段 → 直接返回 `need_rag=false`（跳过 LLM 调用）
3. 构建 TMF 摘要 + 模板字段列表
4. 通过 `DECISION_PROMPT` 注入 rules.md，让 LLM 判断是否需要 RAG
5. LLM 返回 JSON：`{"need_rag": bool, "reasoning": "...", "queries": [...]}`
6. 异常兜底：解析失败时默认 `need_rag=false`

### 5.4 generate_profile 节点 (`src/agent/nodes.py` 中的 `generate_profile` 函数)

- 按 `GENERATION_PROMPT` 将模板字段、TMF 数据、RAG 结果组合成 prompt
- 四个 source 对应四种处理策略：
  - `tmf` → 直接从输入提取
  - `model` → LLM 上下文推理生成
  - `rag` → 基于检索结果填充
  - `rule` → 按 `prompt_instruction` 定义的规则计算
- 输出纯 JSON（自动剥离 markdown 代码块）

---

## 六、配置文件说明

### 6.1 画像模板 (`config/templates/default_profile.json`)

```json
{
  "name": "default",
  "version": "1.0.0",
  "fields": [
    {"name": "offer_id",                "type": "string", "source": "tmf",   "required": true},
    {"name": "target_users",             "type": "array",  "source": "model", "required": true},
    {"name": "competitive_advantages",   "type": "array",  "source": "rag",   "required": false},
    {"name": "price_segment",            "type": "string", "source": "rule",  "required": false}
  ]
}
```

每条字段的 `source` 定义数据来源：
- `tmf`   — 从输入的 TMF JSON 中提取
- `model` — 由 LLM 根据上下文生成
- `rag`   — 通过 RAG 知识库检索填充
- `rule`  — 按 `prompt_instruction` 中的规则计算

### 6.2 RAG 规则 (`config/rules/rag_rules.md`)

定义何时调用 RAG、查询生成规则、结果使用规则。**模型在 rag_decision 节点读取此文件进行自主决策**。

### 6.3 RAG API 配置 (`config/rag_config.yaml`)

```yaml
base_url: "http://localhost:8000/api/v1"
timeout: 30
max_retries: 2
default_top_k: 5
endpoints:
  - name: search
    path: /rag/search
    method: POST
  - name: retrieve
    path: /rag/retrieve
    method: POST
    headers:
      X-Retrieval-Mode: hybrid
```

### 6.4 示例数据 (`data/sample_tmf.json`)

"5G畅享Pro套餐" — 包含 id, name, description, category, price, characteristics, product_specification, bundled_product_offering。

---

## 七、运行方式

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env：
#   DEEPSEEK_API_KEY=sk-xxx          # DeepSeek 平台申请: https://platform.deepseek.com/api_keys
#   OPENAI_API_KEY=sk-xxx            # 与上面相同 (ChatOpenAI 读取此变量)
#   OPENAI_BASE_URL=https://api.deepseek.com/v1
#   LLM_MODEL=deepseek-chat          # 也可用 deepseek-reasoner

# 3. 修改 RAG API 配置（按实际情况）
# 编辑 config/rag_config.yaml

# 4. 运行
python -m src.main --input data/sample_tmf.json --verbose

# 指定输出路径
python -m src.main -i data/sample_tmf.json -o output/result.json
```

> **Note**: 项目通过 `langchain-openai` 的 ChatOpenAI 连接 DeepSeek API。
> DeepSeek 100% 兼容 OpenAI 格式，因此无需修改任何代码，
> 只需配置 `OPENAI_BASE_URL=https://api.deepseek.com/v1` 和 `LLM_MODEL=deepseek-chat`。

---

## 八、扩展指南

| 需求 | 操作 |
|------|------|
| **新增画像模板** | 创建 `config/templates/xxx_profile.json`，调用 `load_profile_template("xxx")` |
| **修改 RAG 规则** | 编辑 `config/rules/rag_rules.md`，模型自动适配 |
| **新增节点** | 在 `src/agent/nodes.py` 添加函数，在 `graph.py` 注册节点和边 |
| **切换 LLM** | 修改 `.env` 中 `LLM_MODEL` 和 `OPENAI_BASE_URL` |
| **切换 RAG 端点** | 修改 `config/rag_config.yaml` 中 endpoints 配置 |
| **流式输出** | 利用 LangGraph 的 `stream()` 模式 |

---

## 九、设计决策记录

1. **LLM 通过闭包注入**：`graph.py` 中 `_rag_decision_wrapper(llm)` 和 `_generate_profile_wrapper(llm)` 将 LLM 实例注入节点，避免节点直接依赖全局 LLM。
2. **配置全部外置**：所有可配置项（模板、规则、API）均通过 YAML/MD 文件管理，代码零修改即可适配不同场景。
3. **条件边实现 RAG 跳过**：`_route_after_decision()` 读取 state.need_rag 进行路由，确保 RAG 不可用时工作流仍可完成。
4. **rag_decision 有安全兜底**：模板中无 `source=rag` 字段时跳过 LLM 调用；LLM 解析失败时默认 `need_rag=false`。
5. **RAG 客户端支持多格式**：`_parse_response()` 处理 3 种常见 API 响应格式，降低与具体 RAG 后端的耦合。
6. **validate_profile 不阻断输出**：即使校验发现错误也保留 `final_output`，由调用方决定是否采纳。
