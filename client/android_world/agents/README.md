# AndroidWorld Agents

## Overview

The agents in this folder are designed to interact with Android devices by
perceiving the device state and taking appropriate actions to accomplish user
goals. The framework uses a hierarchical structure with a base agent class that
specialized agents extend.

## Agents

### M3A (Multimodal Autonomous Agent for Android)

- Featured in the AndroidWorld paper (ICLR 2025)
- Uses both visual and textual data to interact with Android devices
- Capable of understanding screenshots and text descriptions of UI elements
- Uses Set-of-Mark action space

### T3A (Text-only Autonomous Agent for Android)

- Featured in the AndroidWorld paper (ICLR 2025)
- Text-only version of M3A
- Uses only textual representations of UI elements without visual information
- Works with the same action space as M3A

### SeeAct

- Featured in the AndroidWorld paper (ICLR 2025)
- Web-adapted version of the agent from "GPT-4V(ision) is a Generalist Web Agent, if Grounded"
- Uses visual grounding with a two-step reasoning process
- Specialized for interacting with Android interfaces

### Base and Utility Classes

- `base_agent.py`: Abstract base class defining the agent interface
- `agent_utils.py`: Common utility functions used by agents
- `infer.py`: Inference utilities for working with language models
- `*_utils.py`: Agent-specific utility functions

## Usage

Agents implement the `EnvironmentInteractingAgent` interface, providing a
consistent way to:

- Get the device state
- Take actions on the device
- Process feedback after actions
- Track progress through a series of interactions

Each agent has specific initialization requirements but follows the same basic
interaction pattern through the `step()` method.

---

## Privacy Module (`privacy/`)

The `privacy/` subdirectory implements a privacy-aware screen processing layer (V3 architecture) that intercepts screenshots and masks private visual content before the agent sees it.

### Architecture

```
Server (agentPrivacy)          Local (this codebase)
─────────────────────          ──────────────────────────────────────────
/api/v1/detect  ──────────►  screen_context.py  → ScreenContext (page type)
/api/v1/vqa     ◄──────────  privacy_classifier.py → expose / hard_mask
/api/v1/llm     ──────────►  task_extractor.py  → TaskRequirements
/api/v1/ner     ──────────►  image_redactor.py  → redacted pixels
```

### Decision Logic

**Default rule — controls always visible:**
Non-image elements (buttons, text, icons) are always exposed so the agent can navigate.

**Image elements** are decided by matching task requirements against the current page type:

```
For each ContentRequirement(subject, scope) in task_reqs.requirements:
    if current page type matches scope (or scope == "anywhere"):
        subject == "all"  → EXPOSE  (all images on this page type)
        subject is text   → VQA check → EXPOSE if matched
No match → HARD_MASK
```

### Key Files

| File | Role |
|------|------|
| `screen_context.py` | Activity/package name → `ScreenContext` (80+ apps, Activity-level rules) |
| `task_extractor.py` | LLM extracts `List[ContentRequirement(subject, scope)]` from task text |
| `privacy_classifier.py` | Per-detection expose/mask decision + VQA |
| `image_redactor.py` | Applies hard_mask redaction to pixels |
| `server_hooks.py` | Orchestrates the full pipeline + Interactive Disclosure |
| `env_wrapper.py` | Wraps `AsyncEnv.get_state()` to inject privacy processing |
| `hooks.py` | `PrivacyHooks` interface (base class) |

### LLM Output Format (`task_extractor.py`)

The LLM outputs a JSON array. Each entry specifies what visual content is needed and where:

```json
[
  {"subject": "all",      "scope": "map_view"},
  {"subject": "猫猫照片", "scope": "gallery"},
  {"subject": "猫猫照片", "scope": "anywhere"}
]
```

Valid `scope` values match `ScreenContext` enum names plus `"anywhere"` (wildcard).

### Interactive Disclosure

When the agent taps on a masked (`hard_mask`) region, `on_action()` detects the tap, temporarily reveals the original content for that region, and logs the disclosure event to `disclosure_logging/disclosure_log.json`.