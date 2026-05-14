"""
ProfileAgent 主入口

用法:
    python -m src.main --input data/sample_tmf.json

流程:
    1. 加载环境变量和 LLM
    2. 编译 LangGraph Agent
    3. 读取 TMF 输入 JSON
    4. 运行 Agent 工作流
    5. 输出 offer 画像 JSON
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from src.agent.graph import compile_agent

# 加载 .env
_ = load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def get_llm() -> ChatOpenAI:
    """从环境变量创建 LLM 实例"""
    _api_key = os.getenv("OPENAI_API_KEY")
    return ChatOpenAI(
        model=os.getenv("LLM_MODEL", "gpt-4o"),
        api_key=_api_key if _api_key else None,  # type: ignore[arg-type]
        base_url=os.getenv("OPENAI_BASE_URL"),
        temperature=0.1,
    )


def run_agent(
    tmf_path: str,
    output_path: str | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    运行 ProfileAgent，根据 TMF JSON 生成 offer 画像

    Args:
        tmf_path: TMF JSON 文件路径
        output_path: 输出路径（可选，默认输出到 output/ 目录）
        verbose: 是否打印详细日志

    Returns:
        生成的 offer 画像 dict
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # 1. 读取 TMF 输入
    input_path = Path(tmf_path)
    if not input_path.exists():
        raise FileNotFoundError(f"TMF 输入文件不存在: {tmf_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        tmf_data = json.load(f)
    logger.info(f"读取 TMF 输入: {input_path.name}")

    # 2. 初始化 LLM
    llm = get_llm()
    logger.info(f"LLM: {llm.model_name} @ {llm.openai_api_base}")

    # 3. 编译 Agent
    agent = compile_agent(llm)
    logger.info("Agent 编译完成")

    # 4. 运行
    initial_state = {
        "tmf_input": tmf_data,
        "rules_md": "",
        "profile_template": {},
        "rag_api_config": {},
        "messages": [],
        "need_rag": False,
        "rag_queries": [],
        "rag_context": [],
        "profile_output": None,
        "validation_errors": [],
        "final_output": None,
    }

    logger.info("开始执行 Agent 工作流...")
    result = agent.invoke(initial_state)
    logger.info("Agent 工作流执行完成")

    # 5. 输出结果
    final_output = result.get("final_output", result.get("profile_output"))
    errors = result.get("validation_errors", [])

    if errors:
        logger.warning(f"校验发现 {len(errors)} 个问题:")
        for err in errors:
            logger.warning(f"  - {err}")

    # 确定输出路径
    if output_path:
        out_path = Path(output_path)
    else:
        out_dir = Path("output")
        out_dir.mkdir(exist_ok=True)
        stem = input_path.stem
        out_path = out_dir / f"{stem}_profile.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)
    logger.info(f"画像已输出到: {out_path}")

    if final_output is None:
        raise ValueError("画像生成失败：最终输出为空")
    return final_output


def main():
    parser = argparse.ArgumentParser(
        description="ProfileAgent - 运营商套餐 Offer 画像生成 Agent"
    )
    _ = parser.add_argument("--input", "-i", required=True,
                            help="TMF 格式的套餐 JSON 文件路径")
    _ = parser.add_argument("--output", "-o", default=None,
                            help="输出画像 JSON 路径 (默认 output/{input}_profile.json)")
    _ = parser.add_argument("--verbose", "-v", action="store_true",
                            help="启用详细日志")
    args = parser.parse_args()

    try:
        result = run_agent(
            tmf_path=args.input,
            output_path=args.output,
            verbose=args.verbose,
        )
        logger.info(f"\n{'='*60}")
        logger.info("生成结果预览:")
        logger.info(json.dumps(result, ensure_ascii=False, indent=2)[:500])
        sys.exit(0)
    except Exception as e:
        logger.error(f"执行失败: {e}", exc_info=args.verbose)
        sys.exit(1)


if __name__ == "__main__":
    main()
