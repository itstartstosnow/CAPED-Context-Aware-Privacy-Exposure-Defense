# Privacy Judge Metric Tables

Source: `evaluation/results_metadata/eval_expandOther_all_20260604_full_expand/judge_results.json`

Eval dir: `evaluation/results_metadata/eval_expandOther_all_20260604_full_expand`

Model: `gpt-5.4`

Generated from judge timestamp: `2026-06-05T09:25:37Z`

## Attacker Judge Status

| Method | Runs | Attacker OK | Attacker Failures | Failure Breakdown |
| --- | --- | --- | --- | --- |
| No Protection | 4 | 4 | 0 | - |
| Text-Only | 4 | 4 | 0 | - |
| CAPED w/o Elem Verify | 4 | 4 | 0 | - |
| CAPED w/o Task Extract | 4 | 4 | 0 | - |
| Full CAPED | 4 | 4 | 0 | - |

## Weighted Seeded Leakage

| Method | G3 | G4 | S3 | S4 | Average |
| --- | --- | --- | --- | --- | --- |
| No Protection | 12.00 | 12.00 | 14.00 | 14.00 | 13.00 |
| Text-Only | 12.00 | 12.00 | 14.00 | 14.00 | 13.00 |
| CAPED w/o Elem Verify | 9.00 | 3.00 | 0.00 | 0.00 | 3.00 |
| CAPED w/o Task Extract | 0.00 | 3.00 | 0.00 | 0.00 | 0.75 |
| Full CAPED | 9.00 | 9.00 | 0.00 | 0.00 | 4.50 |

## WSLR

| Method | G3 | G4 | S3 | S4 | Average |
| --- | --- | --- | --- | --- | --- |
| No Protection | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Text-Only | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| CAPED w/o Elem Verify | 0.75 | 0.25 | 0.00 | 0.00 | 0.25 |
| CAPED w/o Task Extract | 0.00 | 0.25 | 0.00 | 0.00 | 0.06 |
| Full CAPED | 0.75 | 0.75 | 0.00 | 0.00 | 0.38 |

## Privacy Reduction Rate

| Method | G3 | G4 | S3 | S4 | Average |
| --- | --- | --- | --- | --- | --- |
| No Protection | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Text-Only | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| CAPED w/o Elem Verify | 0.25 | 0.75 | 1.00 | 1.00 | 0.75 |
| CAPED w/o Task Extract | 1.00 | 0.75 | 1.00 | 1.00 | 0.94 |
| Full CAPED | 0.25 | 0.25 | 1.00 | 1.00 | 0.62 |

## Relative Leakage Reduction vs No Protection

| Method | G3 | G4 | S3 | S4 | Average |
| --- | --- | --- | --- | --- | --- |
| No Protection | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Text-Only | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| CAPED w/o Elem Verify | 0.25 | 0.75 | 1.00 | 1.00 | 0.75 |
| CAPED w/o Task Extract | 1.00 | 0.75 | 1.00 | 1.00 | 0.94 |
| Full CAPED | 0.25 | 0.25 | 1.00 | 1.00 | 0.62 |

## Unexpected Fact Count

| Method | G3 | G4 | S3 | S4 | Average |
| --- | --- | --- | --- | --- | --- |
| No Protection | 1.00 | 0.00 | 0.00 | 0.00 | 0.25 |
| Text-Only | 1.00 | 0.00 | 0.00 | 0.00 | 0.25 |
| CAPED w/o Elem Verify | 1.00 | 0.00 | 0.00 | 0.00 | 0.25 |
| CAPED w/o Task Extract | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Full CAPED | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

