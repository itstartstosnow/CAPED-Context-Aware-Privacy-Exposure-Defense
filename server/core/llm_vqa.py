"""core/llm_vqa.py — Qwen3.5-0.8B 多模态 VQA

复用 core/llm.py 的模型单例（同一权重，不重复加载），
新增图像输入能力。

接口：
  llm_vqa_single(image_b64, concept) -> {answer: yes|no, raw: str}
  llm_vqa_multi(image_b64, concepts) -> {present: [str], raw: str}
"""

import base64
import io
import torch
from PIL import Image


def _get_model():
    """确保模型已加载，返回 (model, processor, device)。"""
    import core.llm as llm_mod
    llm_mod._load_llm()
    return llm_mod._LLM_MODEL, llm_mod._LLM_PROCESSOR, llm_mod._LLM_DEVICE


def _build_inputs(processor, device, pil_image, text_prompt):
    """构造 Qwen3-VL 多模态输入 tensor。"""
    from qwen_vl_utils import process_vision_info

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text",  "text": text_prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        return_tensors="pt",
        padding=True,
    )
    return {k: v.to(device) for k, v in inputs.items()}


def _generate(model, processor, inputs, max_new_tokens=32):
    """生成文本，返回新 token 解码后的字符串。"""
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    input_len = inputs["input_ids"].shape[1]
    output_ids = generated[0][input_len:]
    return processor.decode(output_ids, skip_special_tokens=True).strip()


def llm_vqa_single(image_b64: str, concept: str) -> dict:
    """单概念 VQA：Does this image contain <concept>?

    Returns:
        {"answer": "yes"|"no", "raw": str}
    """
    model, processor, device = _get_model()
    if model is None:
        return {"answer": "no", "raw": "model not loaded"}

    try:
        img = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
        prompt = f"Does this image contain {concept}? Answer yes or no."
        inputs = _build_inputs(processor, device, img, prompt)
        raw = _generate(model, processor, inputs, max_new_tokens=16)
        answer = "yes" if "yes" in raw.lower() else "no"
        print(f"[llm_vqa_single] concept='{concept}' → raw='{raw}' → {answer}")
        return {"answer": answer, "raw": raw}
    except Exception as exc:
        print(f"[llm_vqa_single] error: {exc}")
        return {"answer": "no", "raw": f"error: {exc}"}


def llm_vqa_multi(image_b64: str, concepts: list) -> dict:
    """多概念 VQA：逐项 yes/no + few-shot 格式示范（方案 A+C）。

    Prompt 策略：
      - few-shot 例子用纯文字描述另一张假设图片，展示多选合法性
      - 逐项 yes/no 格式（方案A），避免模型只输出一个最显著项
      - 全程中文，减少跨语言指令偏移
      - 解析：逐行匹配 "概念: yes"

    Returns:
        {"present": [concept, ...], "raw": str}
    """
    import re

    model, processor, device = _get_model()
    if model is None:
        return {"present": [], "raw": "model not loaded"}

    try:
        img = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")

        # Build per-item checklist
        checklist = "\n".join(f"- {c}:" for c in concepts)

        prompt = (
            "【格式示例 — 以下是另一张假设图片，仅用于说明回答格式，不是你要分析的图】\n"
            "假设图片内容：一张咖啡馆室内照片，有咖啡杯、笔记本电脑和书，没有汽车和动物。\n"
            "- 咖啡: yes\n"
            "- 汽车: no\n"
            "- 电脑: yes\n"
            "- 书: yes\n"
            "- 动物: no\n"
            "【格式示例结束】\n\n"
            "现在请仔细分析上方提供的实际图片，对下列每一项逐行回答 yes 或 no：\n"
            + checklist
        )

        inputs = _build_inputs(processor, device, img, prompt)
        raw = _generate(model, processor, inputs, max_new_tokens=len(concepts) * 12 + 16)

        # Parse per-line "concept: yes/no"
        present = []
        for c in concepts:
            # Match lines like "- 王者荣耀: yes" or "王者荣耀: yes" (case-insensitive yes)
            pattern = re.escape(c) + r"[\s:：]+yes"
            if re.search(pattern, raw, re.IGNORECASE):
                present.append(c)

        print(f"[llm_vqa_multi] raw='{raw}' → present={present}")
        return {"present": present, "raw": raw}
    except Exception as exc:
        print(f"[llm_vqa_multi] error: {exc}")
        return {"present": [], "raw": f"error: {exc}"}


def _build_inputs_batch(processor, device, pairs: list):
    """Build batched inputs for N (PIL.Image, prompt_str) pairs.

    Uses left-padding so all sequences end at the same position — required for
    decoder-only batch generation so that new tokens start at consistent offsets.
    """
    from qwen_vl_utils import process_vision_info

    all_messages = [
        [{"role": "user", "content": [{"type": "image", "image": img},
                                      {"type": "text",  "text": prompt}]}]
        for img, prompt in pairs
    ]
    texts = [
        processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        for msgs in all_messages
    ]
    # process_vision_info expects a flat list of message dicts
    flat_messages = [msg for msgs in all_messages for msg in msgs]
    image_inputs, _ = process_vision_info(flat_messages)
    inputs = processor(
        text=texts,
        images=image_inputs,
        return_tensors="pt",
        padding=True,
        padding_side="left",  # decoder-only: left-pad so real content is right-aligned
    )
    return {k: v.to(device) for k, v in inputs.items()}


def _generate_batch(model, processor, inputs, max_new_tokens: int = 16) -> list:
    """Run batch generation; return list[str] (one decoded string per item)."""
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    # attention_mask.sum(dim=1) gives the actual (non-padding) input length per item
    input_lengths = inputs["attention_mask"].sum(dim=1)
    return [
        processor.decode(generated[i, input_lengths[i]:], skip_special_tokens=True).strip()
        for i in range(len(input_lengths))
    ]


def llm_vqa_single_batch(pairs: list) -> list:
    """Batch single-concept VQA: one model.generate() call for all pairs.

    Args:
        pairs: list of (image_b64: str, concept: str)

    Returns:
        list of {"answer": "yes"|"no", "raw": str} matching input order.
    """
    model, processor, device = _get_model()
    if model is None:
        return [{"answer": "no", "raw": "model not loaded"}] * len(pairs)

    try:
        decoded = [
            (Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB"),
             f"Does this image contain {concept}? Answer yes or no.")
            for b64, concept in pairs
        ]
        inputs = _build_inputs_batch(processor, device, decoded)
        raws = _generate_batch(model, processor, inputs)

        results = []
        for (_, concept), raw in zip(pairs, raws):
            answer = "yes" if "yes" in raw.lower() else "no"
            print(f"[llm_vqa_batch] concept='{concept}' → raw='{raw}' → {answer}")
            results.append({"answer": answer, "raw": raw})
        return results

    except Exception as exc:
        print(f"[llm_vqa_single_batch] error: {exc}")
        return [{"answer": "no", "raw": f"error: {exc}"}] * len(pairs)
