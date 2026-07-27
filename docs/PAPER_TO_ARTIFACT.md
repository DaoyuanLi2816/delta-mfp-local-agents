# Paper-to-artifact map

| Paper evidence | Released source | Regeneration or verification |
|---|---|---|
| 120-task suite | `data/tasks/local_calibrated_tasks.jsonl` | `tests/test_tasks.py` |
| Calibration table | `data/processed/calibration_results.csv` | `run_local_llm_suite.py --aggregate` |
| Natural regimes | `data/processed/natural_mfp_results.csv` | `run_local_llm_suite.py --natural` |
| Persistent positive control | `data/processed/fault_injection_results.csv` | `run_local_llm_suite.py --faults` |
| Full soft `N=5` headline | `soft_fault_results_N5_full.csv` | `code/run_n5_full.py` |
| `N=2` to `N=5` turnover | `soft_replay_stability_full.csv` | `code/run_n5_full.py` |
| Repair probe and Wilson intervals | `repair_results_N3.csv`, `repair_summary_N3.csv` | `code/run_repair_n3.py` |
| Qwen2.5-14B probe | `cross_model_robustness.csv` | `code/run_cross_model.py` |
| Figures and compact tables | `figures/`, processed CSVs | `code/plotting/make_figures.py` |
| All headline counts | all rows above | `scripts/verify_paper_claims.py` |
| Camera-ready paper | `paper/main.pdf` | checksum in `paper/SHA256SUMS` |

The processed CSVs are the released source of truth for paper numbers. Model
reruns create new observations and are not expected to be byte-identical.
