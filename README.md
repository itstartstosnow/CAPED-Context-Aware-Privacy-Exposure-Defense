# CAPED: Context-Aware Privacy Exposure Defense for Mobile GUI Agents

Official code and evaluation artifact repository for **CAPED** (*Context-Aware Privacy Exposure Defense*), a pre-upload exposure control layer for screenshot-based mobile GUI agents.

## Overview

Screenshot-based mobile GUI agents observe full smartphone screens to operate apps, turning every visual observation into a privacy boundary. During normal task execution, screenshots frequently expose non-target contacts, chat previews, photos, filenames, personalized shopping recommendations, and health cues that are unrelated to the user's instruction (**incidental visual privacy exposure**).

CAPED mediates screenshots and task instructions *before* they cross the local--remote boundary:
1. **Local Task Requirement Extraction**: De-identifies explicit entities via NER and extracts structured task requirements `{app, subject, type}` (`person`, `item`, `date`, `open`, `none`).
2. **Screen Context Classification**: Maps foreground app package and Activity signals to coarse context privacy postures (`Public`, `Privacy`, `Text-Privacy`).
3. **UI Element Parsing**: Parses visible screen elements into coarse `text`, `icon`, and `image` regions via OCR and GUI component detection (UIED + PaddleOCR).
4. **Task-Aware Exposure Resolution**: Combines context prior, task requirements, element modality, semantic text relevance (EmbeddingGemma), and subject/navigation VQA to selectively expose only task-needed content.
5. **Structure-Preserving Redaction**: Masks incidental visual regions with a solid overlay and eye affordance while replacing textual PII with consistent typed placeholders.

---

## Repository Structure

```text
CAPED/
├── server/                        # CAPED model inference server (FastAPI: detect, NER, VQA, LLM, embedding)
│   ├── api/                       # FastAPI routes and schemas (/api/v1/*)
│   ├── core/                      # Model wrappers (UIED+OCR, GLiNER, Qwen3.5, EmbeddingGemma)
│   ├── UIED/                      # UI Element Detection pipeline
│   └── README.md                  # Server API specification & open-source model download links
├── client/                        # AndroidWorld-based evaluation harness & CAPED policy client
│   ├── android_world/             # Base AndroidWorld framework + CAPED agent implementation
│   │   └── agents/
│   │       ├── privacy_guiowl.py  # PrivacyAwareGUIOwl main agent entry (3 hook points)
│   │       └── privacy/           # CAPED policy engine, task extractor, context classifier, redactor, ablations
│   ├── privacy_tasks/             # 28-task Seeded Privacy Benchmark task definitions (AWP & expansion)
│   ├── privacy_seeders/           # Deterministic local privacy data seeders (Files, Gallery, SMS, etc.)
│   ├── setup_mock_app_snapshots.py# Mock privacy app installer & snapshot generator
│   └── run_*.py / run_*.sh        # Evaluation runners for AndroidWorld and Seeded Privacy Benchmark
└── evaluation/                    # Privacy judge, seeded ontology, and paper evaluation tables
    ├── privacy_judge/
    │   ├── privacy_task_registry.json       # Registry of all 28 seeded privacy tasks
    │   ├── task_specs.json                  # Task-required vs. incidental information boundaries
    │   ├── seeded_leakage_specs.json        # Seeded leakage ontology and severity weights (1/2/3)
    │   ├── run_privacy_judge.py             # Two-stage LLM attacker judge (Discovery + Ontology Mapping)
    │   ├── build_paper_privacy_tables.py    # Deterministic WSLR / SPS / Utility table generator
    │   ├── paper_privacy_update_20260606/   # Aggregated paper-ready CSV/JSON/Markdown tables
    │   └── judge_stability_spotcheck_20260610/ # 3-repeat judge stability spot-check artifacts
    └── results_metadata/          # Per-run judge outputs, manual utility records, and AndroidWorld timing tables (raw logs omitted)
```

> **Dataset & Full Trajectories**: Complete screenshot trajectories (28 tasks × 5 methods = 140 runs, 1,223 screenshots), mock app APKs (`SunShop`, `MessengerPlus`, `HealthTrack`), and synthetic seed media assets are hosted in the companion Hugging Face Dataset: [**`startstosnow/CAPED-Context-Aware-Privacy-Exposure-Defense-Benchmark`**](https://huggingface.co/datasets/startstosnow/CAPED-Context-Aware-Privacy-Exposure-Defense-Benchmark).

---

## Quick Start

### 1. Start Remote GUI Agent Model (GUI-Owl-7B via vLLM)

Serve [`mPLUG/GUI-Owl-7B`](https://huggingface.co/mPLUG/GUI-Owl-7B) using an OpenAI-compatible vLLM endpoint (default port `8000`):

```bash
python3 -m vllm.entrypoints.openai.api_server \
    --served-model-name GUI-Owl-7B \
    --model mPLUG/GUI-Owl-7B \
    --gpu-memory-utilization 0.7 \
    --max-model-len 16384 \
    --port 8000 \
    --trust-remote-code \
    --dtype bfloat16 \
    --limit-mm-per-prompt '{"image": 2}'
```

### 2. Start CAPED Privacy Model Server

See [`server/README.md`](server/README.md) for required open-source model weights and setup instructions.

```bash
cd server
pip install -r requirements.txt
AGENT_PRIVACY_GPU=0 python -m uvicorn api.main:app --host 0.0.0.0 --port 8001
```

### 3. Run Client & Evaluations

```bash
cd client
pip install -r requirements.txt
pip install -e .

# Pre-flight environment check (ADB, emulator snapshots, server ports)
python check_ready.py

# For Mock App Suite: place the 3 compiled APKs from the companion Hugging Face Dataset (seed_assets/mock_app_apks/*.apk)
# into client/mock_app_apks_260511/ before running setup_mock_app_snapshots.py or run_mock_privacy_eval.py.
# Run Mock App Privacy Suite (9 tasks x 5 methods)
python run_mock_privacy_eval.py --base_url http://localhost:8000/v1 --privacy_url http://localhost:8001

# Run Local & Connected Privacy Suites
python run_awp_privacy_eval.py --base_url http://localhost:8000/v1 --privacy_url http://localhost:8001
```

### 4. Reproduce Paper Privacy Tables

All pre-computed judge outputs and manual task-success records are included in `evaluation/results_metadata/`. To regenerate the paper tables directly:

```bash
python evaluation/privacy_judge/build_paper_privacy_tables.py
```

---

## Compared Methods

| Paper Name | Runner ID | Description |
|---|---|---|
| **No Protection** | `guiowl` | Raw screenshots sent directly to GUI-Owl-7B |
| **Text-Only** | `text_only` | OCR + NER textual entity replacement without visual region masking |
| **CAPED w/o Element Verification** | `no_elem_verify` | Task extraction + context policy masking without VQA or embedding rescue |
| **CAPED w/o Task Extraction** | `no_task_extract` | Context-level protection without task requirement unlocking |
| **Full CAPED** | `privacy` | Full pipeline: task extraction, context policy, UI parsing, element verification, and redaction |

## License

This project is released for academic research use. Top-level code follows the MIT License ([LICENSE](LICENSE)); the adapted AndroidWorld client harness under `client/` retains its upstream Apache-2.0 License ([client/LICENSE](client/LICENSE)); the bundled UIED component under `server/UIED/` retains its upstream Apache-2.0 license ([server/UIED/LICENSE](server/UIED/LICENSE)); `client/LICENSE` also carries the upstream MIT notice for the bundled MiniWoB HTML assets.
