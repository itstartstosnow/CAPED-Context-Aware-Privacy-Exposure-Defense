# Privacy Judge Metric Tables

Source: `evaluation/results_metadata/eval_awp_all_updated_260512_1744/judge_results.json`

Eval dir: `evaluation/results_metadata/eval_awp_all_updated_260512_1744`

Model: `gpt-5.4`

Generated from judge timestamp: `2026-05-12T14:31:56Z`

## Accuracy

| Method | A1 | A2 | C1 | S1 | S2 | G2 | Average |
| --- | --- | --- | --- | --- | --- | --- | --- |
| No Protection | 1.00 | 0.80 | 0.60 | 0.80 | 1.00 | 0.80 | 0.83 |
| Text-Only | 1.00 | 0.80 | 0.60 | 0.80 | 1.00 | 0.80 | 0.83 |
| CAPED w/o Elem Verify | 0.60 | 0.60 | 0.60 | 0.60 | 0.60 | 0.60 | 0.60 |
| CAPED w/o Task Extract | 0.60 | 0.60 | 0.40 | 0.60 | 0.60 | 0.80 | 0.60 |
| Full CAPED | 1.00 | 0.60 | 0.40 | 0.60 | 0.60 | 0.80 | 0.67 |

## Recall

| Method | A1 | A2 | C1 | S1 | S2 | G2 | Average |
| --- | --- | --- | --- | --- | --- | --- | --- |
| No Protection | 1.00 | 0.50 | 0.00 | 0.50 | 1.00 | 0.50 | 0.58 |
| Text-Only | 1.00 | 0.50 | 0.50 | 0.50 | 1.00 | 0.50 | 0.67 |
| CAPED w/o Elem Verify | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| CAPED w/o Task Extract | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.50 | 0.08 |
| Full CAPED | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.50 | 0.25 |

## Attacker Judge Status

| Method | Runs | Attacker OK | Attacker Failures | Failure Breakdown |
| --- | --- | --- | --- | --- |
| No Protection | 6 | 6 | 0 | - |
| Text-Only | 6 | 6 | 0 | - |
| CAPED w/o Elem Verify | 6 | 6 | 0 | - |
| CAPED w/o Task Extract | 6 | 6 | 0 | - |
| Full CAPED | 6 | 6 | 0 | - |

## Weighted Seeded Leakage

| Method | A1 | A2 | C1 | S1 | S2 | G2 | Average |
| --- | --- | --- | --- | --- | --- | --- | --- |
| No Protection | 11.00 | 5.00 | 0.00 | 11.00 | 14.00 | 3.00 | 7.33 |
| Text-Only | 11.00 | 5.00 | 3.00 | 11.00 | 14.00 | 3.00 | 7.83 |
| CAPED w/o Elem Verify | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| CAPED w/o Task Extract | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 3.00 | 0.50 |
| Full CAPED | 11.00 | 0.00 | 0.00 | 0.00 | 0.00 | 3.00 | 2.33 |

## WSLR

| Method | A1 | A2 | C1 | S1 | S2 | G2 | Average |
| --- | --- | --- | --- | --- | --- | --- | --- |
| No Protection | 0.79 | 0.36 | 0.00 | 0.79 | 1.00 | 0.25 | 0.53 |
| Text-Only | 0.79 | 0.36 | 0.25 | 0.79 | 1.00 | 0.25 | 0.57 |
| CAPED w/o Elem Verify | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| CAPED w/o Task Extract | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.25 | 0.04 |
| Full CAPED | 0.79 | 0.00 | 0.00 | 0.00 | 0.00 | 0.25 | 0.17 |

## Privacy Reduction Rate

| Method | A1 | A2 | C1 | S1 | S2 | G2 | Average |
| --- | --- | --- | --- | --- | --- | --- | --- |
| No Protection | 0.21 | 0.64 | 1.00 | 0.21 | 0.00 | 0.75 | 0.47 |
| Text-Only | 0.21 | 0.64 | 0.75 | 0.21 | 0.00 | 0.75 | 0.43 |
| CAPED w/o Elem Verify | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| CAPED w/o Task Extract | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.75 | 0.96 |
| Full CAPED | 0.21 | 1.00 | 1.00 | 1.00 | 1.00 | 0.75 | 0.83 |

## Relative Leakage Reduction vs No Protection

| Method | A1 | A2 | C1 | S1 | S2 | G2 | Average |
| --- | --- | --- | --- | --- | --- | --- | --- |
| No Protection | 0.00 | 0.00 | N/A | 0.00 | 0.00 | 0.00 | 0.00 |
| Text-Only | 0.00 | 0.00 | N/A | 0.00 | 0.00 | 0.00 | 0.00 |
| CAPED w/o Elem Verify | 1.00 | 1.00 | N/A | 1.00 | 1.00 | 1.00 | 1.00 |
| CAPED w/o Task Extract | 1.00 | 1.00 | N/A | 1.00 | 1.00 | 0.00 | 0.80 |
| Full CAPED | 0.00 | 1.00 | N/A | 1.00 | 1.00 | 0.00 | 0.60 |

## Unexpected Fact Count

| Method | A1 | A2 | C1 | S1 | S2 | G2 | Average |
| --- | --- | --- | --- | --- | --- | --- | --- |
| No Protection | 0.00 | 0.00 | 4.00 | 0.00 | 0.00 | 0.00 | 0.67 |
| Text-Only | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.17 |
| CAPED w/o Elem Verify | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| CAPED w/o Task Extract | 0.00 | 2.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.33 |
| Full CAPED | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

