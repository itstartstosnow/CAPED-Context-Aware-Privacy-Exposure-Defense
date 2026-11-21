# AndroidWorld 116-Task General Utility Summary

| Method | Completed Tasks | Task Success Rate (TSR) |
| --- | ---: | ---: |
| No Protection (GUI-Owl-7B) | 77 / 116 | 66.4% |
| Full CAPED + GUI-Owl-7B | 64 / 116 | 55.2% |

Per-task binary outcomes (`0` / `1`) for all 116 tasks are provided in `task_outcomes_116.csv` (58 both succeed, 33 both fail, 19 regressions, 6 improvements). Raw screenshot trajectories and run logs for the 116 general-utility tasks (~1.3 GB) are omitted from the public release footprint; all 140 trajectories for the 28-task Seeded Privacy Benchmark are published in the companion Hugging Face Dataset. The `log` and `run_dir` columns in `timing_per_task.csv` / `timing_per_segment.csv` are provenance pointers to those unpublished logs.
