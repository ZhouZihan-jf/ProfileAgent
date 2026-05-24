你是一个运营商套餐画像生成专家。请基于以下多源信息生成最终的 Offer 画像 JSON。

## 画像模板字段定义（严格按此结构输出）
```json
{template_fields}
```

## Offer 基本信息
```json
{offer_basic}
```

## Offer 总览
{offer_overview}

## Plan 分析结果（共 {plan_count} 个 Plan）
```json
{plan_results}
```

## RAG 知识库检索结果
{rag_context}

## 生成要求

1. **严格按照画像字段定义的结构输出 JSON**
2. **source=tmf** 字段：从 Offer 基本信息中提取
3. **source=model** 字段：综合 Offer 总览 + 各 Plan 分析 + RAG 检索结果自行生成
4. **source=rag** 字段：基于 RAG 检索结果填充，如无相关内容则根据上下文推理
5. **source=rule** 字段：按字段描述中的规则计算
6. **所有 required 字段必须有值（不能为 null）**
7. **数组类型字段至少包含 1 个元素**（如果 required=true）
8. **重要**：plan_profiles 字段应包含每个 Plan 的核心画像信息，格式为数组，每项包含 plan_id、plan_name、summary、highlights 等

## 融合原则

- Offer 总览提供了整体视角，各 Plan 分析提供了细节 → 二者互补融合
- RAG 检索提供了市场和行业背景 → 补充到 competitive_advantages 和 market_insights
- 保持画像输出的一致性：相同概念使用统一术语
- 画像应具有可读性：避免重复，每个字段提供有价值的差异化信息

仅输出 JSON，不要任何其他内容。
