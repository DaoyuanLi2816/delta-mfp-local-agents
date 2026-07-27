# Reproducibility

The artifact supports three distinct levels. Choose the lowest level that
answers your question.

## Level 1: verify the released evidence

No GPU, model, API, or network access:

```bash
python -m pip install -e ".[dev]"
python -m pytest
python scripts/verify_paper_claims.py
```

This checks the 120-task suite shape, retained trace counts, every headline
regime count, the `N=2` to `N=5` stability result, the 14B probe, selected
repair cells, and the final paper checksum.

## Level 2: regenerate derived figures and tables

No model inference:

```bash
python code/plotting/make_figures.py \
  --out reproduced/figures \
  --tables-out reproduced/tables
```

Outputs are isolated under `reproduced/`; committed result CSVs and
camera-ready figures are not overwritten.

## Level 3: rerun local-model observations

Install and start Ollama, then pull:

```bash
ollama pull qwen2.5:7b
ollama pull llama3.1:8b
ollama pull qwen2.5:14b
ollama serve
```

Camera-ready replay and repair runs:

```bash
python code/run_n5_full.py
python code/run_repair_n3.py
python code/run_cross_model.py --model qwen2.5:14b --per-subtype 4
```

Full task generation and observation collection:

```bash
python code/run_local_llm_suite.py --write-tasks
python code/run_local_llm_suite.py --models qwen2.5:7b llama3.1:8b qwen2.5:14b --inventory
python code/run_local_llm_suite.py --calibrate --models qwen2.5:7b \
  --interfaces raw_json json_parse_repair tool_compiler \
  --families calendar refund file_email inventory \
  --calibration-seeds 5 --per-family-difficulty 10
python code/run_local_llm_suite.py --natural --natural-target 100 \
  --natural-seeds-per-cell 4 --replay-n 3 --p-fail 0.6 --delta 0.3
python code/run_local_llm_suite.py --faults --fault-target 40 \
  --replay-n 2 --p-fail 0.6 --delta 0.3 --skip-fault-repairs
python code/run_soft_faults.py --target 50 --replay-n 1 --max-attempts 120
```

The original run used Windows, Python 3.10, Ollama, and an RTX 4080 with 16 GB
VRAM. Exact generations can vary with Ollama version, quantization, and local
model build. Preserve new raw outputs separately until they have been
compared with the released artifact.

## Statistical scope

- Natural replay: `N=3`.
- Persistent positive control: `N=2`.
- Soft camera-ready audit: `N=5`.
- Repair probe: `N=3`, at most five trace IDs per cell, Wilson 95% intervals.
- Cross-model probe: `N=3`, 24 balanced soft scenarios.

Do not compare repair point estimates without the cell sizes and intervals.
Do not collapse unstable/no-Delta into successful localization.
