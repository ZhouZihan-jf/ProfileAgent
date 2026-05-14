"""
RAG RESTful API 客户端

支持模型自主评估后按需调用，包含请求构建、响应解析和错误处理。
"""

import logging
from typing import Any
import httpx

from src.config.config import RAGAPIConfig, RAGEndpoint

logger = logging.getLogger(__name__)


class RAGClient:
    """
    RAG 知识检索客户端

    通过 RESTful API 与知识库交互，模型在 decision 节点根据 rules.md
    自行决定是否调用以及如何构建查询参数。
    """

    def __init__(self, config: RAGAPIConfig):
        self.config = config
        self._client: httpx.Client | None = None

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            headers = {"Content-Type": "application/json"}
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"
            self._client = httpx.Client(
                base_url=self.config.base_url,
                headers=headers,
                timeout=httpx.Timeout(self.config.timeout),
            )
        return self._client

    def get_endpoint(self, name: str) -> RAGEndpoint:
        """根据名称获取端点配置"""
        for ep in self.config.endpoints:
            if ep.name == name:
                return ep
        raise ValueError(
            f"未找到端点 '{name}'，可用端点: {[e.name for e in self.config.endpoints]}"
        )

    def search(
        self,
        query: str,
        top_k: int | None = None,
        endpoint_name: str = "search",
        **extra_params,
    ) -> list[dict[str, Any]]:
        """
        调用 RAG 检索接口

        Args:
            query: 检索查询词
            top_k: 返回文档数，默认使用配置值
            endpoint_name: 使用的端点名称
            **extra_params: 额外的请求参数

        Returns:
            检索到的文档列表，每条包含 content 和 metadata
        """
        endpoint = self.get_endpoint(endpoint_name)
        top_k = top_k or self.config.default_top_k

        payload = {
            "query": query,
            "top_k": top_k,
            **extra_params,
        }

        # 合并端点自定义 headers
        headers = {**endpoint.headers}

        for attempt in range(self.config.max_retries + 1):
            try:
                method = endpoint.method.upper()
                if method == "GET":
                    response = self.client.get(
                        endpoint.path, params=payload, headers=headers
                    )
                else:
                    response = self.client.post(
                        endpoint.path, json=payload, headers=headers
                    )
                response.raise_for_status()
                return self._parse_response(response.json())

            except httpx.TimeoutException:
                logger.warning(f"RAG 请求超时 (尝试 {attempt + 1}): {query[:50]}...")
                if attempt == self.config.max_retries:
                    raise
            except httpx.HTTPStatusError as e:
                logger.error(f"RAG API 错误: {e.response.status_code} - {e.response.text}")
                raise

        return []

    def batch_search(
        self, queries: list[str], top_k: int | None = None
    ) -> dict[str, list[dict[str, Any]]]:
        """批量检索多个查询"""
        results = {}
        for q in queries:
            try:
                results[q] = self.search(q, top_k=top_k)
            except Exception:
                logger.warning(f"批量检索失败 - 查询: {q[:50]}...")
                results[q] = []
        return results

    def _parse_response(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        """
        解析 RAG API 响应为标准格式

        尝试解析多种常见 API 响应格式，确保输出统一：
        每条文档包含: {content, score, metadata}
        """
        # 格式 1: { "results": [{ "content": "...", "score": 0.9, "metadata": {...} }] }
        if "results" in data:
            return data["results"]

        # 格式 2: { "documents": [{ "text": "...", "similarity": 0.9 }] }
        if "documents" in data:
            return [
                {
                    "content": d.get("text", d.get("content", "")),
                    "score": d.get("score", d.get("similarity", 0)),
                    "metadata": d.get("metadata", d.get("meta", {})),
                }
                for d in data["documents"]
            ]

        # 格式 3: { "data": [...] }
        if "data" in data and isinstance(data["data"], list):
            return [
                {
                    "content": str(d),
                    "score": 0,
                    "metadata": {},
                }
                for d in data["data"]
            ]

        logger.warning(f"未知的 RAG 响应格式: {list(data.keys())}")
        return []

    def close(self):
        """关闭 HTTP 客户端"""
        if self._client:
            self._client.close()
            self._client = None
