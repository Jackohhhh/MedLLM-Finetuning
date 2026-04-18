"""训练时根据 model_name_or_path 与 LoRA/量化模式推导 experiment_name、output_dir、prediction_output_dir。"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def hf_model_slug(model_name_or_path: str) -> str:
    """Hub id 或本地路径中的分隔符转为目录安全片段，例如 Qwen/Qwen2-7B-Instruct -> Qwen_Qwen2-7B-Instruct。"""
    s = (model_name_or_path or "").strip()
    if not s:
        raise ValueError("model_name_or_path is empty.")
    return s.replace("\\", "/").replace("/", "_").replace(" ", "_")


def train_mode_suffix(config: dict[str, Any]) -> str:
    """根据最终 config 区分 full / lora / qlora（及 lora8bit）。"""
    m = config.get("model") or {}
    lora_on = bool((m.get("lora") or {}).get("enabled", False))
    if not lora_on:
        return "full"
    if m.get("load_in_4bit"):
        return "qlora"
    if m.get("load_in_8bit"):
        return "lora8bit"
    return "lora"


def apply_train_run_paths(config: dict[str, Any]) -> None:
    """在 CLI 覆盖已合并后调用，写入 experiment_name 与输出目录。"""
    model_id = (config.get("model") or {}).get("model_name_or_path")
    if not model_id:
        raise ValueError(
            "训练需要基座模型：请在 CLI 传入 --model-name-or-path，或在 configs/models/*.yaml 中设置 model_name_or_path。"
        )
    base = hf_model_slug(str(model_id))
    mode = train_mode_suffix(config)
    name = f"{base}_{mode}"
    config["experiment_name"] = name
    config["output_dir"] = str(Path("outputs/checkpoints") / name)
    config["prediction_output_dir"] = str(Path("outputs/predictions") / name)
