"""Pydantic 模型定义 - 符合 API specification 规范"""
from pydantic import BaseModel
from typing import Optional, Dict, Any, List


class HealthResponse(BaseModel):
    """健康检查响应模型"""
    status: str
    models_loaded: bool
    gpu_available: bool


class NerProcessResponse(BaseModel):
    """NER 处理响应模型 - 直接返回字段，符合 client 期望

    参考 API specification:
    - anonymized_text: 脱敏后的文本
    - ner_mapping: 映射表
    - entities: 检测到的实体列表
    """
    anonymized_text: str
    ner_mapping: Dict[str, Any]
    entities: List[Dict[str, Any]]


# ── V3 新增 schemas ────────────────────────────────────────────────────────────

class DetectResponse(BaseModel):
    """POST /api/v1/detect 响应 — 原始检测结果，不含隐私决策"""
    detections: List[Dict[str, Any]]
    labeled_image: Optional[str] = None   # Base64 PNG overlay (optional)
    server_log: Optional[str] = None


class VqaResponse(BaseModel):
    """POST /api/v1/vqa 响应"""
    answer: str                 # "yes" | "no"
    confidence: float = 1.0


class LlmRequest(BaseModel):
    """POST /api/v1/llm 请求体"""
    prompt: str
    max_tokens: int = 256
    stop: list = []  # stop sequences applied via post-processing


class LlmResponse(BaseModel):
    """POST /api/v1/llm 响应"""
    text: str



class EmbedRequest(BaseModel):
    texts: list


class EmbedResponse(BaseModel):
    embeddings: list
    dim: int


# ── LLM VQA schemas (Qwen multimodal) ────────────────────────────────────────

class LlmVqaSingleRequest(BaseModel):
    """POST /api/v1/llm_vqa/single 请求体"""
    image_crop: str          # base64 PNG
    concept: str             # 要查询的概念，e.g. WeChat app icon


class LlmVqaSingleResponse(BaseModel):
    """POST /api/v1/llm_vqa/single 响应"""
    answer: str              # yes | no
    raw: str                 # 模型原始输出


class LlmVqaMultiRequest(BaseModel):
    """POST /api/v1/llm_vqa/multi 请求体"""
    image_crop: str          # base64 PNG
    concepts: List[str]      # 候选概念列表


class LlmVqaMultiResponse(BaseModel):
    """POST /api/v1/llm_vqa/multi 响应"""
    present: List[str]       # 出现在图中的概念
    raw: str                 # 模型原始输出


# ── LLM VQA Batch schemas ─────────────────────────────────────────────────────

class LlmVqaBatchItem(BaseModel):
    """Single (image, concept) pair in a batch request."""
    image_crop: str   # base64 PNG
    concept: str


class LlmVqaBatchRequest(BaseModel):
    """POST /api/v1/llm_vqa/single_batch 请求体"""
    items: List[LlmVqaBatchItem]


class LlmVqaBatchResultItem(BaseModel):
    """One VQA result in a batch response."""
    answer: str   # "yes" | "no"
    raw: str      # model raw output


class LlmVqaBatchResponse(BaseModel):
    """POST /api/v1/llm_vqa/single_batch 响应"""
    results: List[LlmVqaBatchResultItem]
