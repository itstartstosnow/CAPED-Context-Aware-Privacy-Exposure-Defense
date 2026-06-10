# Runtime Overhead Summary (AndroidWorld Live Loop)

Weighted per-step averages over 1,394 measured steps across the AndroidWorld evaluation:

| Component | Mean Time per Step (s) | Description |
| --- | ---: | --- |
| `t_observation_fetch_s` | 3.097 | `env.get_state()` observation fetch |
| `t_vlm_inference_s` | 2.571 | GUI-Owl-7B inference for action prediction |
| `t_action_exec_and_settle_s` | 2.302 | ADB action execution and settle wait |
| **Original Agent Subtotal** | **7.970** | Observation + VLM inference + actuation |
| `t_foreground_activity_s` | 0.085 | Foreground activity/package query |
| `t_detect_s` | 1.811 | UI element & OCR detection |
| `t_local_policy_and_prep_s` | 0.311 | Local policy resolution & crop preparation |
| `t_vqa_s` | 0.088 | Batched crop VQA verification |
| `t_embed_s` | 0.001 | Text semantic embedding relevance |
| `t_redact_s` | 0.028 | Redacted screenshot rendering |
| `t_action_check_s` | 0.000 | Pre-action disclosure check |
| **CAPED Online Subtotal** | **2.325** | Online per-step CAPED components (22.6% of attributed 10.295s component sum; 21.5% of total step wall time) |
| **Total Step Time (`t_step_total_s`)** | **10.835** | End-to-end measured step wall time (remaining 0.540s is inter-hook harness/IPC bookkeeping) |

One-time task requirement extraction (`task_extract_cached_app_list_timing/summary.md`): mean `5.716s` (median `5.634s`) across 115 tasks.
