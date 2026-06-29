# Δ-MFP: Local Tool-Use Agent Failure Diagnosis

Curated reproducibility release for the camera-ready paper:

**Before the Fall: Δ Minimal Failing Prefixes for Local Tool-Use Agent Failures**
Daoyuan Li, University of Minnesota Twin Cities. FAGEN @ ICML 2026 (workshop, non-archival).

Δ-MFP (Delta Minimal Failing Prefix) is a counterfactual-replay diagnostic for
stateful tool-use agents. For a failed trace it compares the replay failure
probability `p_k` from a prefix against the failure probability `p_0` from the
initial state, classifying each failure into **prefix-0**, **nontrivial Δ-MFP**
(cascade), **unstable / no-Δ**, or **irreversible / costly**. All experiments run
on a single local GPU.

This repository is curated: it contains only the code that reproduces the paper,
the input traces and tasks, the result CSVs the paper actually uses, and the
final figures. No model weights, secrets, or absolute paths.

## Hardware & models
- NVIDIA RTX 4080, 16 GB VRAM; Windows / PowerShell (Python is OS-portable).
- [Ollama](https://ollama.com) local server (`http://localhost:11434`), called via stdlib `urllib`.
- Python 3.10+.

| Model | Params | Quant. | Role |
|---|---|---|---|
| `qwen2.5:7b`  | 7.6B  | Q4_K_M | main model for Δ-MFP / replay |
| `llama3.1:8b` | 8.0B  | Q4_K_M | calibration + part of the soft set |
| `qwen2.5:14b` | 14.8B | Q4_K_M | calibration + cross-model robustness check |

Decoding: `temperature=0.35`, `num_predict=64` (one syntax-repair retry at `0.1`).
Fixed parameters: `p_fail=0.6`, `delta=0.3`. Replay budgets: natural `N=3`,
persistent positive control `N=2`, **soft `N=5`**, **repair `N=3`**, cross-model `N=3`.

## Setup
```bash
pip install -r requirements.txt          # matplotlib, numpy
ollama pull qwen2.5:7b && ollama pull llama3.1:8b && ollama pull qwen2.5:14b
ollama serve
```

## Reproduce the camera-ready experiments (from the included traces; no regeneration)
```bash
python code/run_n5_full.py        # Full N=5 soft audit, all 50 traces (~2.5 h on RTX 4080)
python code/run_repair_n3.py      # Repair re-evaluation at N=3, same 9 trace-ids x 10 methods (~1 h)
python code/run_cross_model.py --model qwen2.5:14b --per-subtype 4   # Cross-model robustness, 24 cases (~30 min)
python code/plotting/make_figures.py    # Regenerate figures + the compact repair table
```
Outputs land in `data/processed/` and `figures/`.

## Reproduce the full pipeline from scratch (many GPU-hours)
```bash
python code/run_local_llm_suite.py --write-tasks
python code/run_local_llm_suite.py --models qwen2.5:7b llama3.1:8b qwen2.5:14b --inventory
python code/run_local_llm_suite.py --calibrate --models qwen2.5:7b \
  --interfaces raw_json json_parse_repair tool_compiler \
  --families calendar refund file_email inventory --calibration-seeds 5 --per-family-difficulty 10
python code/run_local_llm_suite.py --natural --natural-target 100 --natural-seeds-per-cell 4 --replay-n 3 --p-fail 0.6 --delta 0.3
python code/run_local_llm_suite.py --faults --fault-target 40 --replay-n 2 --p-fail 0.6 --delta 0.3 --skip-fault-repairs
python code/run_soft_faults.py --target 50 --replay-n 1 --max-attempts 120
python code/run_experiment.py     # controlled mechanism suite (appendix sanity check)
```

## Which artifact produces which paper element
| Paper element | Script | Output |
|---|---|---|
| Calibration table | `run_local_llm_suite.py --aggregate` | `data/processed/calibration_results.csv` |
| Natural-failure regimes | `run_local_llm_suite.py --natural` | `data/processed/natural_mfp_results.csv` |
| Persistent positive control | `run_local_llm_suite.py --faults` | `data/processed/fault_injection_results.csv` |
| Soft regimes (full N=5 headline) | `run_n5_full.py` | `soft_fault_results_N5_full.csv`, `soft_replay_stability_full.csv` |
| Repair (N=3) | `run_repair_n3.py` | `repair_results_N3.csv`, `repair_summary_N3.csv` |
| Cross-model robustness | `run_cross_model.py` | `cross_model_robustness.csv` |
| Figures + repair table | `plotting/make_figures.py` | `figures/*.pdf`, repair compact table |

## Layout
```
code/        agents/ benchmarks/ replay/ repairs/ analysis/ plotting/ tests/ + run_*.py drivers
data/tasks/  local_calibrated_tasks.jsonl (120 tasks)
data/traces/ soft_fault_traces.jsonl, fault_injected_traces.jsonl, natural_failed_traces.jsonl
data/processed/  the result CSVs the paper uses
figures/     final paper figures
prompts/     interface prompts, task generation, labeling schema
router_rules.md   rule order for the Predicted repair router
```

## License
MIT (`LICENSE`), © 2026 Daoyuan Li.
