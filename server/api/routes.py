from pydantic import BaseModel as _PBM
"""FastAPI 路由定义 — V3 架构

保留端点：
  GET  /health           — 健康检查（不变）
  POST /ner/process      — NER 脱敏（不变）

新增端点：
  POST /detect           — UIED 检测，返回原始 JSON（不做隐私决策）
  POST /vqa              — Florence-2 VQA yes/no 查询
  POST /llm              — Qwen3.5-0.8B / 2B 文本补全

删除端点：
  POST /process          — v2 重量级端点（隐私决策移至本地）
"""
import contextlib
import io
import json
import time
import tempfile
import os

from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional

from api.schemas import (
    HealthResponse, NerProcessResponse,
    DetectResponse, VqaResponse, LlmRequest, LlmResponse,
)
from api.dependencies import get_controller

router = APIRouter()


# ── Health ────────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
async def health_check():
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        return HealthResponse(status="healthy", models_loaded=True, gpu_available=gpu_available)
    except Exception:
        return HealthResponse(status="unhealthy", models_loaded=False, gpu_available=False)


# ── NER ───────────────────────────────────────────────────────────────────────

@router.post("/ner/process", response_model=NerProcessResponse)
async def ner_process(
    text: str = Form(..., description="要脱敏的文本"),
    ner_mapping: Optional[str] = Form(None, description="之前的 NER 映射 JSON 字符串"),
):
    """NER 文本脱敏 — 无状态：每次请求创建独立 anonymizer 实例，不共享跨任务状态。"""
    from core.anonymization import TextDeAnonymizer
    # GLOBAL_NER_MODEL 已在 startup 预加载，此处不重新加载模型，代价极小
    anonymizer = TextDeAnonymizer()
    try:
        if ner_mapping:
            anonymizer.load_mapping_from_dict(json.loads(ner_mapping))

        anonymized_text, meta = anonymizer.anonymize_text(text)
        current_mapping = {
            "mapping_table": dict(anonymizer.mapping_table),
            "entity_counters": dict(anonymizer.entity_counters),
            "reverse_mapping": anonymizer.get_reverse_mapping(),
        }
        return NerProcessResponse(
            anonymized_text=anonymized_text,
            ner_mapping=current_mapping,
            entities=meta.get("entities", []),
        )
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid ner_mapping JSON: {exc}")
    except Exception:
        import traceback; traceback.print_exc()
        return NerProcessResponse(
            anonymized_text=text,
            ner_mapping={"mapping_table": {}, "entity_counters": {}, "reverse_mapping": {}},
            entities=[],
        )


# ── Detect ────────────────────────────────────────────────────────────────────

@router.post("/detect", response_model=DetectResponse)
async def detect(
    image: UploadFile = File(..., description="截图文件"),
    debug: bool = Form(False, description="True 时返回 server_log"),
):
    """UIED 检测端点 — 仅返回原始检测 JSON，不做隐私决策。

    Response detections format:
      [{"type": "text"|"icon"|"image", "bbox": [x1,y1,x2,y2],
        "content": str|None, "source": str, "image_label_id": int}, ...]
    """
    from core.uiedDetection import detectUI

    image_bytes = await image.read()

    # Save to temp file (detectUI requires a file path)
    tmp_dir = tempfile.mkdtemp(prefix="detect_api_")
    img_path = os.path.join(tmp_dir, "input.png")
    try:
        with open(img_path, "wb") as f:
            f.write(image_bytes)

        if debug:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                result = detectUI(img_path, debug=False)
            server_log = buf.getvalue()
        else:
            result = detectUI(img_path, debug=False)
            server_log = None

        detections = result.get("detections", [])
        labeled_b64 = result.get("labeled_image_base64")

        return DetectResponse(
            detections=detections,
            labeled_image=labeled_b64,
            server_log=server_log,
        )
    finally:
        import shutil; shutil.rmtree(tmp_dir, ignore_errors=True)


# ── LLM ───────────────────────────────────────────────────────────────────────

@router.post("/llm_08b", response_model=LlmResponse)
async def llm_08b_endpoint(body: LlmRequest):
    """Qwen3.5-0.8B 文本补全接口（保留用于对比测试）。

    Request body (JSON):
      {"prompt": "<full prompt>", "max_tokens": 256, "stop": [...]}

    Response:
      {"text": "<generated text>"}
    """
    from core.llm import llm_generate

    text = llm_generate(body.prompt, max_tokens=body.max_tokens, stop=body.stop or [])
    return LlmResponse(text=text)


@router.post("/llm_2b", response_model=LlmResponse)
async def llm_2b_endpoint(body: LlmRequest):
    """Qwen3.5-2B 文本补全接口。

    Request body (JSON):
      {"prompt": "<full prompt>", "max_tokens": 256, "stop": [...]}

    Response:
      {"text": "<generated text>"}
    """
    from core.llm_2b import llm_2b_generate

    text = llm_2b_generate(body.prompt, max_tokens=body.max_tokens, stop=body.stop or [])
    return LlmResponse(text=text)

# -- Embed Gemma --------------------------------------------------------------

@router.post("/embed_gemma", response_model=EmbedResponse)
async def embed_gemma_endpoint(body: EmbedRequest):
    from core.embed_gemma import embed_texts_gemma
    if not body.texts:
        return EmbedResponse(embeddings=[], dim=768)
    embeddings = embed_texts_gemma(body.texts)
    if not embeddings:
        raise HTTPException(status_code=503, detail="EmbeddingGemma model unavailable")
    return EmbedResponse(embeddings=embeddings, dim=len(embeddings[0]))


# ── LLM VQA (Qwen3.5-0.8B multimodal) ───────────────────────────────────────

from api.schemas import LlmVqaSingleRequest, LlmVqaSingleResponse, LlmVqaMultiRequest, LlmVqaMultiResponse


@router.post("/llm_vqa/single", response_model=LlmVqaSingleResponse)
async def llm_vqa_single_endpoint(body: LlmVqaSingleRequest):
    """Qwen3.5-0.8B 单概念图像 VQA。

    Request: {"image_crop": "<base64 PNG>", "concept": "WeChat app icon"}
    Response: {"answer": "yes"|"no", "raw": "<model output>"}
    """
    from core.llm_vqa import llm_vqa_single
    result = llm_vqa_single(body.image_crop, body.concept)
    return LlmVqaSingleResponse(answer=result["answer"], raw=result["raw"])


@router.post("/llm_vqa/multi", response_model=LlmVqaMultiResponse)
async def llm_vqa_multi_endpoint(body: LlmVqaMultiRequest):
    """Qwen3.5-0.8B 多概念图像 VQA，一次询问哪些概念出现在图中。

    Request: {"image_crop": "<base64 PNG>", "concepts": ["WeChat", "calendar", "map"]}
    Response: {"present": ["WeChat"], "raw": "<model output>"}
    """
    from core.llm_vqa import llm_vqa_multi
    result = llm_vqa_multi(body.image_crop, body.concepts)
    return LlmVqaMultiResponse(present=result["present"], raw=result["raw"])


# ── LLM VQA Batch endpoint ────────────────────────────────────────────────────

from api.schemas import LlmVqaBatchRequest, LlmVqaBatchResponse, LlmVqaBatchResultItem

_VQA_SERVER_BATCH_SIZE = 8  # Max items per single model.generate() call


@router.post("/llm_vqa/single_batch", response_model=LlmVqaBatchResponse)
async def llm_vqa_single_batch_endpoint(body: LlmVqaBatchRequest):
    """Qwen3.5-0.8B 批量单概念图像 VQA — 一次推理同时处理 N 个 (image, concept) 对。

    Request: {"items": [{"image_crop": "<base64>", "concept": "游戏"}, ...]}
    Response: {"results": [{"answer": "yes"|"no", "raw": "..."}, ...]}

    Server auto-chunks if len(items) > _VQA_SERVER_BATCH_SIZE.
    """
    from core.llm_vqa import llm_vqa_single_batch
    pairs = [(item.image_crop, item.concept) for item in body.items]
    all_results = []
    for i in range(0, len(pairs), _VQA_SERVER_BATCH_SIZE):
        chunk = pairs[i:i + _VQA_SERVER_BATCH_SIZE]
        all_results.extend(llm_vqa_single_batch(chunk))
    return LlmVqaBatchResponse(
        results=[LlmVqaBatchResultItem(**r) for r in all_results]
    )
