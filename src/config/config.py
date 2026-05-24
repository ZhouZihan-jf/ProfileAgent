"""
配置层 - 数据模型 + 文件加载器

所有配置从 YAML/JSON/Markdown 文件读取，运行时通过 Pydantic 校验。
"""

import json
import os
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field


# ==============================================================
#  数据模型
# ==============================================================

class ProfileField(BaseModel):
    """画像字段定义"""
    name: str = Field(description="字段名")
    type: str = Field(default="string", description="字段类型: string, number, array, object")
    description: str = Field(default="", description="字段描述/生成指引")
    required: bool = Field(default=False, description="是否必填")
    default: Any = Field(default=None, description="默认值")
    source: str = Field(default="model", description="数据来源: model(LLM生成)|tmf(从输入提取)|rag(从RAG检索)|rule(规则计算)")
    prompt_instruction: str = Field(default="", description="给LLM的生成提示")


class ProfileTemplate(BaseModel):
    """画像模板 - 定义输出 JSON 的结构"""
    name: str = Field(description="模板名称")
    version: str = Field(default="1.0.0")
    description: str = Field(default="")
    fields: list[ProfileField] = Field(description="画像字段列表")


class RAGEndpoint(BaseModel):
    """RAG API 端点配置"""
    name: str = Field(description="端点名称，如 search / retrieve")
    path: str = Field(description="API 路径")
    method: str = Field(default="POST", description="HTTP 方法")
    headers: dict[str, str] = Field(default_factory=dict, description="额外请求头")


class RAGAPIConfig(BaseModel):
    """RAG RESTful API 整体配置"""
    base_url: str = Field(description="API base URL")
    api_key: Optional[str] = Field(default=None, description="API 密钥")
    endpoints: list[RAGEndpoint] = Field(description="可用端点列表")
    timeout: int = Field(default=30, description="请求超时(秒)")
    max_retries: int = Field(default=2, description="API 调用最大重试次数")
    default_top_k: int = Field(default=5, description="默认返回文档数")
    min_score: float = Field(default=0.3, description="检索结果最低相关性分数阈值")
    rag_loop_max: int = Field(default=2, description="RAG 充分性检查最大重试轮次")


class TMFOffer(BaseModel):
    """TMF 套餐 Offer 简化模型"""
    id: str = Field(description="Offer ID")
    name: str = Field(description="套餐名称")
    description: Optional[str] = Field(default=None, description="套餐描述")
    category: Optional[str] = Field(default=None, description="套餐类别")
    price: Optional[float] = Field(default=None, description="价格")
    characteristics: Optional[dict[str, Any]] = Field(default=None, description="套餐属性")
    product_specification: Optional[dict[str, Any]] = Field(default=None, description="产品规格")
    bundled_product_offering: Optional[list[dict[str, Any]]] = Field(default=None, description="捆绑产品")


# ==============================================================
#  文件加载器
# ==============================================================

def _get_config_dir() -> Path:
    """获取 config 目录路径 (项目根目录下的 config/)"""
    return Path(__file__).parent.parent.parent / "config"


def load_profile_template(template_name: str = "default") -> ProfileTemplate:
    """
    加载画像模板配置

    Args:
        template_name: 模板名称，对应 config/templates/{name}_profile.json
    """
    template_path = _get_config_dir() / "templates" / f"{template_name}_profile.json"
    if not template_path.exists():
        raise FileNotFoundError(
            f"模板文件不存在: {template_path}\n"
            f"请创建 JSON 配置文件，参考 config/templates/default_profile.json 示例"
        )
    with open(template_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    fields = [ProfileField(**field) for field in data.get("fields", [])]
    return ProfileTemplate(
        name=data.get("name", template_name),
        version=data.get("version", "1.0.0"),
        description=data.get("description", ""),
        fields=fields,
    )


def load_rag_config() -> RAGAPIConfig:
    """加载 RAG API 配置"""
    config_path = _get_config_dir() / "rag_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"RAG 配置文件不存在: {config_path}\n"
            f"请创建 config/rag_config.yaml"
        )
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    endpoints = [RAGEndpoint(**ep) for ep in data.get("endpoints", [])]
    return RAGAPIConfig(
        base_url=data["base_url"],
        api_key=data.get("api_key"),
        endpoints=endpoints,
        timeout=data.get("timeout", 30),
        max_retries=data.get("max_retries", 2),
        default_top_k=data.get("default_top_k", 5),
        min_score=data.get("min_score", 0.3),
        rag_loop_max=data.get("rag_loop_max", 2),
    )


def load_rules() -> str:
    """加载 RAG 使用规则 Markdown 文件"""
    rules_path = _get_config_dir() / "rules" / "rag_rules.md"
    if not rules_path.exists():
        raise FileNotFoundError(
            f"规则文件不存在: {rules_path}\n"
            f"请创建 config/rules/rag_rules.md"
        )
    with open(rules_path, "r", encoding="utf-8") as f:
        return f.read()


def load_prompt(prompt_name: str) -> str:
    """
    加载 prompt 模板文件

    Args:
        prompt_name: prompt 名称，对应 config/prompts/{name}.md
    """
    prompt_path = _get_config_dir() / "prompts" / f"{prompt_name}.md"
    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Prompt 文件不存在: {prompt_path}\n"
            f"请在 config/prompts/ 目录下创建 {prompt_name}.md"
        )
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def validate_tmf_input(data: dict[str, Any]) -> dict[str, Any]:
    """校验 TMF 输入 JSON 基本结构"""
    required_fields = ["id", "name"]
    missing = [k for k in required_fields if k not in data]
    if missing:
        raise ValueError(f"TMF 输入缺少必填字段: {missing}")
    return data
