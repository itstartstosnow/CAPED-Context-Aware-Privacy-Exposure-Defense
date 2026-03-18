"""core/embed_gemma.py — google/embeddinggemma-300m Embedding 服务（全局单例）

接口：
  embed_texts_gemma(texts: list[str]) -> list[list[float]]
    返回每条文本的 L2-normalized embedding 向量（维度 768）。

模型：google/embeddinggemma-300m
  - Google DeepMind 专为向量搜索设计的双向注意力（Encoder 架构）
  - sentence-transformers 加载，支持 normalize_embeddings
"""

import os
from sentence_transformers import SentenceTransformer

_EMBED_MODEL = None
_LOCAL_CACHE = os.path.join(os.environ.get("CAPED_MODEL_DIR", "./models"), "embeddinggemma-300m")
_MODEL_NAME = "google/embeddinggemma-300m"


def _load_embed():
    global _EMBED_MODEL
    if _EMBED_MODEL is not None:
        return

    load_path = _LOCAL_CACHE if os.path.exists(os.path.join(_LOCAL_CACHE, "config.json")) else _MODEL_NAME

    print(f"[embed_gemma] Loading EmbeddingGemma from {load_path}...")
    try:
        _EMBED_MODEL = SentenceTransformer(load_path)
        print("[embed_gemma] ✓ EmbeddingGemma-300m loaded")
    except Exception as exc:
        print(f"[embed_gemma] Failed to load: {exc}")
        _EMBED_MODEL = None


def embed_texts_gemma(texts: list) -> list:
    """Return L2-normalized embeddings for a list of strings.

    Args:
        texts: List of strings to embed.

    Returns:
        List of float lists, shape [len(texts), 768].
        Returns empty list on model load failure.
    """
    _load_embed()
    if _EMBED_MODEL is None:
        print("[embed_gemma] Model not available; returning empty embeddings.")
        return []

    try:
        embeddings = _EMBED_MODEL.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()
    except Exception as exc:
        print(f"[embed_gemma] Inference failed: {exc}")
        return []


def preload():
    """Preload embedding model at server startup."""
    _load_embed()
