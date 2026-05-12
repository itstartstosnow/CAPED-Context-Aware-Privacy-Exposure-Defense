# Privacy Judge Metric Tables

Source: `evaluation/results_metadata/eval_mock_all_updated_260511_1838/judge_results.json`

Eval dir: `evaluation/results_metadata/eval_mock_all_updated_260511_1838`

Model: `gpt-5.4`

Generated from judge timestamp: `2026-05-12T03:24:21Z`

## Accuracy

| Method | M1 | M1b | M1c | M2 | M2b | M2c | M3 | M3b | M3c | Average |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| No Protection | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Text-Only | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| CAPED w/o Elem Verify | 0.60 | 0.60 | 1.00 | 0.80 | 0.80 | 0.80 | 0.80 | 1.00 | 0.60 | 0.78 |
| CAPED w/o Task Extract | 0.60 | 0.60 | 0.60 | 0.60 | 0.40 | 0.60 | 0.60 | 0.60 | 0.60 | 0.58 |
| Full CAPED | 0.60 | 0.60 | 1.00 | 0.80 | 0.80 | 0.80 | 0.80 | 1.00 | 0.80 | 0.80 |

## Recall

| Method | M1 | M1b | M1c | M2 | M2b | M2c | M3 | M3b | M3c | Average |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| No Protection | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Text-Only | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| CAPED w/o Elem Verify | 0.00 | 0.00 | 1.00 | 0.50 | 0.50 | 0.50 | 0.50 | 1.00 | 0.00 | 0.44 |
| CAPED w/o Task Extract | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Full CAPED | 0.00 | 0.00 | 1.00 | 0.50 | 0.50 | 0.50 | 0.50 | 1.00 | 0.50 | 0.50 |

## Attacker Judge Status

| Method | Runs | Attacker OK | Attacker Failures | Failure Breakdown |
| --- | --- | --- | --- | --- |
| No Protection | 9 | 9 | 0 | - |
| Text-Only | 9 | 9 | 0 | - |
| CAPED w/o Elem Verify | 9 | 9 | 0 | - |
| CAPED w/o Task Extract | 9 | 9 | 0 | - |
| Full CAPED | 9 | 9 | 0 | - |

## Weighted Seeded Leakage

| Method | M1 | M1b | M1c | M2 | M2b | M2c | M3 | M3b | M3c | Average |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| No Protection | 7.00 | 7.00 | 7.00 | 19.00 | 19.00 | 19.00 | 13.00 | 13.00 | 6.00 | 12.22 |
| Text-Only | 7.00 | 7.00 | 7.00 | 19.00 | 19.00 | 19.00 | 13.00 | 13.00 | 6.00 | 12.22 |
| CAPED w/o Elem Verify | 1.00 | 0.00 | 7.00 | 2.00 | 0.00 | 0.00 | 5.00 | 5.00 | 0.00 | 2.22 |
| CAPED w/o Task Extract | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Full CAPED | 1.00 | 0.00 | 7.00 | 1.00 | 2.00 | 1.00 | 5.00 | 5.00 | 0.00 | 2.44 |

## WSLR

| Method | M1 | M1b | M1c | M2 | M2b | M2c | M3 | M3b | M3c | Average |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| No Protection | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Text-Only | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| CAPED w/o Elem Verify | 0.14 | 0.00 | 1.00 | 0.11 | 0.00 | 0.00 | 0.38 | 0.38 | 0.00 | 0.22 |
| CAPED w/o Task Extract | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Full CAPED | 0.14 | 0.00 | 1.00 | 0.05 | 0.11 | 0.05 | 0.38 | 0.38 | 0.00 | 0.24 |

## Privacy Reduction Rate

| Method | M1 | M1b | M1c | M2 | M2b | M2c | M3 | M3b | M3c | Average |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| No Protection | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Text-Only | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| CAPED w/o Elem Verify | 0.86 | 1.00 | 0.00 | 0.89 | 1.00 | 1.00 | 0.62 | 0.62 | 1.00 | 0.78 |
| CAPED w/o Task Extract | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Full CAPED | 0.86 | 1.00 | 0.00 | 0.95 | 0.89 | 0.95 | 0.62 | 0.62 | 1.00 | 0.76 |

## Relative Leakage Reduction vs No Protection

| Method | M1 | M1b | M1c | M2 | M2b | M2c | M3 | M3b | M3c | Average |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| No Protection | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Text-Only | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| CAPED w/o Elem Verify | 0.86 | 1.00 | 0.00 | 0.89 | 1.00 | 1.00 | 0.62 | 0.62 | 1.00 | 0.78 |
| CAPED w/o Task Extract | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Full CAPED | 0.86 | 1.00 | 0.00 | 0.95 | 0.89 | 0.95 | 0.62 | 0.62 | 1.00 | 0.76 |

## Unexpected Fact Count

| Method | M1 | M1b | M1c | M2 | M2b | M2c | M3 | M3b | M3c | Average |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| No Protection | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Text-Only | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| CAPED w/o Elem Verify | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 0.00 | 0.22 |
| CAPED w/o Task Extract | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Full CAPED | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 0.00 | 0.22 |

