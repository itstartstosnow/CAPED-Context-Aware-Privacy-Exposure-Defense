# CAPED Privacy Model Inference Server

FastAPI server providing raw model inference endpoints (`/api/v1/*`) for CAPED's local policy engine:
- UI element detection & OCR (`/api/v1/detect`)
- PII named-entity recognition & de-identification (`/api/v1/ner/process`)
- Task requirement extraction & multimodal VQA (`/api/v1/llm_08b`, `/api/v1/llm_2b`, `/api/v1/llm_vqa/*`)
- Semantic text relevance embedding (`/api/v1/embed_gemma`)

All privacy policy resolution, context classification, and image redaction decisions are executed on the client side (`client/android_world/agents/privacy/`).

---

## Required Open-Source Models

The LLM/VLM weights (Qwen, EmbeddingGemma, GLiNER, PaddleOCR) are not redistributed in this repository: by default the server loads them from `$CAPED_MODEL_DIR` (defaults to `./models/`) and falls back to downloading from the official Hugging Face Hub IDs listed below. The converted UIED element classifier (`cnn_elements.onnx`, Apache-2.0) is included in-tree under `server/UIED/cnn/` so the detection pipeline works without a separate download.

| Component | Model Name / Hugging Face ID | Expected Local Path (`$CAPED_MODEL_DIR/...`) | Purpose |
|---|---|---|---|
| **Task Extractor & VQA (Default)** | [`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B) | `./models/Qwen3.5-0.8B` | Local task requirement extraction (`{app, subject, type}`) and batched crop VQA |
| **Task Extractor (2B Variant)** | [`Qwen/Qwen3.5-2B`](https://huggingface.co/Qwen/Qwen3.5-2B) | `./models/Qwen3.5-2B` | Optional 2B parameter model for task extraction comparison |
| **Semantic Relevance Embedding** | [`google/embeddinggemma-300m`](https://huggingface.co/google/embeddinggemma-300m) | `./models/embeddinggemma-300m` | 768-d multilingual embeddings for element-to-subject cosine similarity |
| **NER De-identification** | [`urchade/gliner_multi-v2.1`](https://huggingface.co/urchade/gliner_multi-v2.1) | Auto-cached by `gliner` / Hugging Face | Multilingual PII entity detection & placeholder mapping |
| **OCR Engine** | [PaddleOCR (PP-OCRv3)](https://github.com/PaddlePaddle/PaddleOCR) | Auto-downloaded by `paddleocr` | Screen text detection and recognition |
| **UI Component Classifier** | [UIED CNN (ONNX)](https://github.com/MulongXie/UIED) | `./UIED/cnn/cnn_elements.onnx` | Graphic-element classifier for UIED regions. Converted from UIED's `cnn-rico-1.h5` (Apache-2.0) and included in this repository; `onnx2torch` loads it at startup. |

> **Remote GUI Agent Model**: The base GUI agent [`mPLUG/GUI-Owl-7B`](https://huggingface.co/mPLUG/GUI-Owl-7B) runs as a separate OpenAI-compatible vLLM server (see root `README.md`).

---

## Running the Server

```bash
pip install -r requirements.txt

# Optional: point to a local directory containing pre-downloaded Hugging Face models
export CAPED_MODEL_DIR=./models

# Launch on GPU 0, port 8001
AGENT_PRIVACY_GPU=0 python -m uvicorn api.main:app --host 0.0.0.0 --port 8001
```

---

## API Endpoints (`/api/v1`)

| Method | Endpoint | Request Format | Response Format | Description |
|---|---|---|---|---|
| `GET` | `/api/v1/health` | — | `{status, models_loaded, gpu_available}` | Server and GPU health check |
| `POST` | `/api/v1/detect` | Multipart form: `image` (PNG), `debug` (bool) | `{detections: [{type, bbox, content, source, image_label_id}], labeled_image, server_log}` | UIED + PaddleOCR element detection |
| `POST` | `/api/v1/ner/process` | Form: `text`, optional `ner_mapping` (JSON str) | `{anonymized_text, ner_mapping, entities}` | Stateless NER de-identification with consistent placeholder mapping |
| `POST` | `/api/v1/llm_08b` | JSON: `{prompt, max_tokens, stop}` | `{text}` | Text completion using Qwen3.5-0.8B |
| `POST` | `/api/v1/llm_2b` | JSON: `{prompt, max_tokens, stop}` | `{text}` | Text completion using Qwen3.5-2B |
| `POST` | `/api/v1/llm_vqa/single` | JSON: `{image_crop, concept}` | `{answer: "yes"\|"no", raw}` | Single-crop VQA verification |
| `POST` | `/api/v1/llm_vqa/multi` | JSON: `{image_crop, concepts: [str, ...]}` | `{present: [str, ...], raw}` | Multi-concept crop VQA (which candidate concepts appear) |
| `POST` | `/api/v1/llm_vqa/single_batch` | JSON: `{items: [{image_crop, concept}, ...]}` | `{results: [{answer, raw}, ...]}` | Batched crop VQA verification for subject & navigation checks |
| `POST` | `/api/v1/embed_gemma` | JSON: `{texts: [str, ...]}` | `{embeddings: [[float, ...]], dim: 768}` | L2-normalized EmbeddingGemma-300m vectors |
