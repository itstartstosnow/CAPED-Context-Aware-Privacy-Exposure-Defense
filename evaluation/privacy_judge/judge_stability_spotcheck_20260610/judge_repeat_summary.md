# Judge Stability Spot-Check Summary

Generated: 2026-06-10T19:24:05Z

Sample seed: `20260610`

Sample size: `28` trajectories, `3` repeats each

Judge model/API: `gpt-5.4` / `responses`

Prompt construction, judge calls, seeded mapping validation, and WSLR computation reuse `evaluation/privacy_judge/run_privacy_judge.py`.

## Overall

| Scope | # trajectories | # incidental labels | Item flip rate | Mean WSLR std | Max WSLR range |
| --- | --- | --- | --- | --- | --- |
| all ok repeats | 28 | 136 | 0.0074 | 0.0048 | 0.2308 |

## Sample Coverage

| Suite | Method | # trajectories |
| --- | --- | --- |
| AndroidWorld-style local apps | guiowl | 1 |
| AndroidWorld-style local apps | no_elem_verify | 3 |
| AndroidWorld-style local apps | no_task_extract | 2 |
| AndroidWorld-style local apps | privacy | 2 |
| AndroidWorld-style local apps | text_only | 2 |
| Mock apps | guiowl | 2 |
| Mock apps | no_elem_verify | 1 |
| Mock apps | no_task_extract | 2 |
| Mock apps | privacy | 2 |
| Mock apps | text_only | 4 |
| Real connected apps | guiowl | 1 |
| Real connected apps | no_elem_verify | 1 |
| Real connected apps | no_task_extract | 1 |
| Real connected apps | privacy | 1 |
| Real connected apps | text_only | 3 |

## Suite And Method Breakdown

| Suite | Method | # trajectories | # incidental labels | Item flip rate | Mean WSLR std | Max WSLR range |
| --- | --- | --- | --- | --- | --- | --- |
| AndroidWorld-style local apps | guiowl | 1 | 4 | 0.0000 | 0.0000 | 0.0000 |
| AndroidWorld-style local apps | no_elem_verify | 3 | 14 | 0.0000 | 0.0000 | 0.0000 |
| AndroidWorld-style local apps | no_task_extract | 2 | 9 | 0.0000 | 0.0000 | 0.0000 |
| AndroidWorld-style local apps | privacy | 2 | 9 | 0.0000 | 0.0000 | 0.0000 |
| AndroidWorld-style local apps | text_only | 2 | 10 | 0.0000 | 0.0000 | 0.0000 |
| Mock apps | guiowl | 2 | 13 | 0.0000 | 0.0000 | 0.0000 |
| Mock apps | no_elem_verify | 1 | 2 | 0.0000 | 0.0000 | 0.0000 |
| Mock apps | no_task_extract | 2 | 12 | 0.0000 | 0.0000 | 0.0000 |
| Mock apps | privacy | 2 | 10 | 0.1000 | 0.0666 | 0.2308 |
| Mock apps | text_only | 4 | 22 | 0.0000 | 0.0000 | 0.0000 |
| Real connected apps | guiowl | 1 | 5 | 0.0000 | 0.0000 | 0.0000 |
| Real connected apps | no_elem_verify | 1 | 4 | 0.0000 | 0.0000 | 0.0000 |
| Real connected apps | no_task_extract | 1 | 4 | 0.0000 | 0.0000 | 0.0000 |
| Real connected apps | privacy | 1 | 5 | 0.0000 | 0.0000 | 0.0000 |
| Real connected apps | text_only | 3 | 13 | 0.0000 | 0.0000 | 0.0000 |

## Method Ordering

| Scope | Aggregation | Method ordering by mean WSLR | Method mean WSLR JSON |
| --- | --- | --- | --- |
| all_sampled | repeat_1 | text_only > guiowl > privacy > no_elem_verify > no_task_extract | {"guiowl": 0.6428571428571428, "no_elem_verify": 0.15, "no_task_extract": 0.0, "privacy": 0.25, "text_only": 0.7142857142857143} |
| all_sampled | repeat_2 | text_only > guiowl > privacy > no_elem_verify > no_task_extract | {"guiowl": 0.6428571428571428, "no_elem_verify": 0.15, "no_task_extract": 0.0, "privacy": 0.20384615384615384, "text_only": 0.7142857142857143} |
| all_sampled | repeat_3 | text_only > guiowl > privacy > no_elem_verify > no_task_extract | {"guiowl": 0.6428571428571428, "no_elem_verify": 0.15, "no_task_extract": 0.0, "privacy": 0.20384615384615384, "text_only": 0.7142857142857143} |
| all_sampled | majority_vote | text_only > guiowl > privacy > no_elem_verify > no_task_extract | {"guiowl": 0.6428571428571428, "no_elem_verify": 0.15, "no_task_extract": 0.0, "privacy": 0.20384615384615384, "text_only": 0.7142857142857143} |
| successful_only | repeat_1 | guiowl > text_only > privacy > no_task_extract > no_elem_verify | {"guiowl": 0.8571428571428571, "no_elem_verify": 0.0, "no_task_extract": 0.0, "privacy": 0.21634615384615385, "text_only": 0.7589285714285714} |
| successful_only | repeat_2 | guiowl > text_only > privacy > no_task_extract > no_elem_verify | {"guiowl": 0.8571428571428571, "no_elem_verify": 0.0, "no_task_extract": 0.0, "privacy": 0.15865384615384615, "text_only": 0.7589285714285714} |
| successful_only | repeat_3 | guiowl > text_only > privacy > no_task_extract > no_elem_verify | {"guiowl": 0.8571428571428571, "no_elem_verify": 0.0, "no_task_extract": 0.0, "privacy": 0.15865384615384615, "text_only": 0.7589285714285714} |
| successful_only | majority_vote | guiowl > text_only > privacy > no_task_extract > no_elem_verify | {"guiowl": 0.8571428571428571, "no_elem_verify": 0.0, "no_task_extract": 0.0, "privacy": 0.15865384615384615, "text_only": 0.7589285714285714} |

## Interpretation Notes

- This is a random spot-check over existing judged-ok trajectories, not a full re-judging of all benchmark runs.
- Method ordering is computed only within the sampled spot-check set.
- Human agreement is intentionally skipped for this run.
