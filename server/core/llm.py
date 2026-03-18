"""core/llm.py — Qwen3.5-0.8B 纯文本补全服务（全局单例）

接口：
  llm_generate(prompt: str, max_tokens: int = 256) -> str
    返回生成文本（不含输入 prompt 部分）
"""

import os
import torch
from transformers import Qwen3_5ForConditionalGeneration, AutoProcessor

_LLM_MODEL = None
_LLM_PROCESSOR = None
_LLM_DEVICE = None

_MODEL_NAME = "Qwen/Qwen3.5-0.8B"
_LOCAL_CACHE = os.path.join(os.environ.get("CAPED_MODEL_DIR", "./models"), "Qwen3.5-0.8B")


def _load_llm():
    global _LLM_MODEL, _LLM_PROCESSOR, _LLM_DEVICE
    if _LLM_MODEL is not None:
        return

    _LLM_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    load_path = _LOCAL_CACHE if os.path.exists(os.path.join(_LOCAL_CACHE, "config.json")) else _MODEL_NAME

    print(f"[llm] Loading Qwen3.5-0.8B from {load_path} on {_LLM_DEVICE}...")
    try:
        _LLM_PROCESSOR = AutoProcessor.from_pretrained(load_path)
        _LLM_MODEL = Qwen3_5ForConditionalGeneration.from_pretrained(
            load_path,
            torch_dtype=torch.float16 if "cuda" in _LLM_DEVICE else torch.float32,
            device_map=_LLM_DEVICE,
        )
        print("[llm] ✓ Qwen3.5-0.8B loaded")
    except Exception as exc:
        print(f"[llm] Failed to load LLM: {exc}")
        _LLM_MODEL = None
        _LLM_PROCESSOR = None


def llm_generate(prompt: str, max_tokens: int = 256, stop: list = None) -> str:
    """Generate text continuation for the given prompt.

    Args:
        prompt: Full prompt string (few-shot examples + task).
        max_tokens: Maximum number of new tokens to generate.

    Returns:
        Generated text (excluding the input prompt).
    """
    _load_llm()
    if _LLM_MODEL is None or _LLM_PROCESSOR is None:
        print("[llm] Model not available; returning empty string.")
        return ""

    try:
        # Text-only input (no images)
        inputs = _LLM_PROCESSOR(text=prompt, return_tensors="pt").to(_LLM_DEVICE)
        input_len = inputs["input_ids"].shape[1]

        generated = _LLM_MODEL.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=_LLM_PROCESSOR.tokenizer.eos_token_id,
        )
        # Return only the newly generated tokens
        output_ids = generated[0][input_len:]
        text = _LLM_PROCESSOR.decode(output_ids, skip_special_tokens=True)
        # Apply stop sequences via post-processing
        if stop:
            for seq in stop:
                idx = text.find(seq)
                if idx != -1:
                    text = text[:idx]
        return text.strip()
    except Exception as exc:
        print(f"[llm] Inference failed: {exc}")
        return ""


def preload():
    """Preload the LLM model at server startup."""
    _load_llm()
