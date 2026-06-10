# Task Utility Scores

> **Note**: The 28 seeded privacy tasks use custom instructions and stopping conditions (e.g., stopping at share sheets or inspecting specific items) that are outside AndroidWorld's built-in 116-task evaluators. Because `eval_summary.json` records the built-in AndroidWorld evaluator's automated verdict—which cannot score these custom instructions or stopping conditions—this file records the authoritative trajectory-level utility scores (`1` = full completion, `0.5` = partial credit, `0` = failure) used in the paper.

| Score | Run |
| --- | --- |
| 1 | guiowl__g3_gallery_share_target_photo |
| 1 | guiowl__g4_gallery_open_target_screenshot |
| 1 | guiowl__s3_sms_search_target_thread |
| 1 | guiowl__s4_sms_compose_to_target_contact |
| 0 | no_elem_verify__g3_gallery_share_target_photo |
| 1 | no_elem_verify__g4_gallery_open_target_screenshot |
| 1 | no_elem_verify__s3_sms_search_target_thread |
| 1 | no_elem_verify__s4_sms_compose_to_target_contact |
| 0 | no_task_extract__g3_gallery_share_target_photo |
| 0 | no_task_extract__g4_gallery_open_target_screenshot |
| 1 | no_task_extract__s3_sms_search_target_thread |
| 0 | no_task_extract__s4_sms_compose_to_target_contact |
| 1 | privacy__g3_gallery_share_target_photo |
| 1 | privacy__g4_gallery_open_target_screenshot |
| 1 | privacy__s3_sms_search_target_thread |
| 1 | privacy__s4_sms_compose_to_target_contact |
| 1 | text_only__g3_gallery_share_target_photo |
| 1 | text_only__g4_gallery_open_target_screenshot |
| 1 | text_only__s3_sms_search_target_thread |
| 1 | text_only__s4_sms_compose_to_target_contact |
