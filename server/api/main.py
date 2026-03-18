"""FastAPI 应用入口 — V3"""
import os
os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('USE_JAX', '0')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', os.environ.get('AGENT_PRIVACY_GPU', '0'))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router as api_router

app = FastAPI(
    title="Privacy Protection API — V3",
    description="V3: detect / vqa / llm / ner endpoints",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")


@app.on_event("startup")
async def startup_event():
    """预加载所有模型"""
    print("[API] V3 startup: preloading models...")

    # 1. NER (GLiNER)
    from api.dependencies import get_controller
    controller = get_controller()
    _ = controller.anonymizer.anonymize_text("preload NER")
    print("[API] ✓ GLiNER NER ready")

    # 2. UIED + Florence (for detect + vqa)
    from core.uiedDetection import preload_models
    preload_models()
    print("[API] ✓ UIED + Florence ready")

    # 3. Qwen3.5-0.8B LLM（task extraction + VQA，当前生产模型）
    from core.llm import preload as llm_08b_preload
    llm_08b_preload()
    print("[API] ✓ Qwen3.5-0.8B LLM ready")
    # 如需切换到 2B，注释上面三行并取消下面注释：
    # from core.llm_2b import preload as llm_2b_preload
    # llm_2b_preload()

    # 4. EmbeddingGemma (for /embed_gemma endpoint)
    from core.embed_gemma import preload as gemma_preload
    gemma_preload()
    print("[API] ✓ EmbeddingGemma-300m ready")

    print("[API] All models loaded ✓")


@app.get("/")
async def root():
    return {"message": "Privacy Protection API V3", "version": "3.0.0", "docs": "/docs"}
