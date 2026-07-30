# Delta-MFP for Local Tool-Use Agents

**Locating when a failed local-agent trajectory first enters a reproducible
failure regime.**

[![CI](https://github.com/DaoyuanLi2816/delta-mfp-local-agents/actions/workflows/ci.yml/badge.svg)](https://github.com/DaoyuanLi2816/delta-mfp-local-agents/actions/workflows/ci.yml)
[![Paper](https://img.shields.io/badge/paper-OpenReview-blue)](https://openreview.net/forum?id=KAA8FR6fEq)
[![Poster](https://img.shields.io/badge/poster-PNG-6f42c1)](paper/poster.png)
[![Code license](https://img.shields.io/badge/code-MIT-green.svg)](LICENSE)

> **ICML 2026 workshop paper (non-archival).**
> *Before the Fall: Delta Minimal Failing Prefixes for Local Tool-Use Agent
> Failures* was accepted at the Workshop on Failure Modes in Agentic AI
> (FAGEN).

Delta-MFP is a counterfactual-replay diagnostic for **stateful tool-use
agents**. Given a failed trace, it restores saved prefixes, samples new
continuations, and distinguishes failures that already reproduce from the
initial state from failures associated with a later trajectory state. The
repository contains the diagnostic implementation, a 120-task local testbed,
released traces and results, and no-GPU artifact verification.

[Paper](https://openreview.net/forum?id=KAA8FR6fEq) ·
[Poster](paper/poster.png) ·
[Final PDF](paper/main.pdf) ·
[Artifact map](docs/PAPER_TO_ARTIFACT.md) ·
[Reproducibility guide](docs/REPRODUCIBILITY.md)

## At a glance

| Dimension | Project at a glance |
|---|---|
| **Research question** | Did the failure already reproduce from the initial state, or did the trajectory enter a later failure basin? |
| **Diagnostic** | Restore each saved prefix, estimate replay failure probability, and classify the resulting failure regime |
| **Testbed** | 120 deterministic tasks with natural failures, persistent positive controls, and soft perturbations |
| **Execution** | Local Ollama inference on a single GPU, with released traces and a fully GPU-free verification path |

## Poster

The poster condenses the Delta-MFP definition, replay phase diagram,
failure-regime profiles, and the implications of finite replay budgets into
one page.

[<img src="paper/poster.png" alt="Before the Fall research poster" width="100%">](paper/poster.png)

*Open the image for the full-resolution 4096 × 2304 PNG.*

## Key findings

| Probe | Released result | Interpretation |
|---|---:|---|
| Natural failures, `N=3` | 13/25 nontrivial, 5/25 prefix-0, 7/25 unstable | Natural local-agent failures occupy distinct replay regimes. |
| Persistent injections, `N=2` | 40/40 localize at the injected prefix | Positive control for snapshot restoration and replay. |
| Soft perturbations, full `N=5` | 7/50 nontrivial, 22/50 prefix-0, 21/50 unstable/no-Delta | Quiet perturbations often do not create stable replay basins. |
| Soft `N=2` → `N=5` | 37/50 keep their regime; only 1/7 earlier nontrivial localizations survives | Small replay budgets can change per-trace attribution even when aggregate counts look stable. |
| Qwen2.5-14B probe, `N=3` | 5/24 nontrivial, 19/24 unstable, 0 prefix-0 | The unstable-dominant pattern persists in the larger-model probe. |

The published counts are executable claims:

```bash
python scripts/verify_paper_claims.py
```

The verifier uses only the bundled tasks, traces, CSVs, and final-paper hash.
It needs no GPU, Ollama server, model weights, or network access.

## How Delta-MFP works

For a failed trace, let `p_k` be the estimated failure probability when replay
starts from saved prefix `k`, and let `p_0` be the corresponding probability
from the initial state. With the paper's `p_fail=0.6` and `delta=0.3`, the
diagnostic reports:

| Regime | Meaning |
|---|---|
| **Prefix-0** | The initial state already fails with high probability; inspect interface, scaffolding, or capability. |
| **Nontrivial Delta-MFP** | A later prefix crosses `p_fail` and raises failure probability by at least `delta`; inspect that transition. |
| **Unstable / no-Delta** | Finite replay does not support a stable attribution; report uncertainty instead of forcing a location. |
| **Irreversible / costly** | Replay alone is not an adequate repair model; use rollback, compensation, or another intervention. |

```text
failed trace
    |
    v
restore prefixes k = 0 ... T
    |
    v
sample independent continuations
    |
    v
estimate p_k and compare with p_0
    |
    v
report the full failure-regime profile
```

## Quick start

Python 3.10+ is supported.

```bash
python -m pip install -e ".[dev]"
python -m pytest
python scripts/verify_paper_claims.py
```

To regenerate the camera-ready figures and compact tables from the committed
result CSVs:

```bash
python code/plotting/make_figures.py \
  --out reproduced/figures \
  --tables-out reproduced/tables
```

## Reproduce the paper

The released artifact supports two different goals:

1. **Verify the published evidence without a model.** Run the test suite,
   claim verifier, and figure generator against the committed tasks, traces,
   and result tables.
2. **Rerun model observations locally.** Install Ollama and use the exact
   drivers, decoding settings, and replay budgets in
   [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

The original experiments used local Ollama inference on one NVIDIA RTX 4080
with 16 GB VRAM:

| Model | Quantization | Role |
|---|---|---|
| `qwen2.5:7b` | Q4_K_M | main model and repeated replay |
| `llama3.1:8b` | Q4_K_M | calibration and part of the soft set |
| `qwen2.5:14b` | Q4_K_M | calibration and cross-model probe |

Decoding uses temperature `0.35`, `num_predict=64`, and one syntax-repair retry
at `0.1`. The reproducibility guide records the exact commands, budgets,
runtime scope, and the distinction between verifying released evidence and
rerunning stochastic model observations.

## Repository map

```text
code/           agent interfaces, simulator, replay, repairs, and analysis
data/tasks/     120 deterministic tasks
data/traces/    retained natural, persistent, and soft failed traces
data/processed/ released result CSVs used by the paper
figures/        camera-ready figure exports
scripts/        no-GPU artifact, claim, and release verification
tests/          deterministic replay, metric, task, and claim tests
docs/           results, reproduction levels, and paper-to-artifact map
paper/          camera-ready PDF, poster, citation, and checksum
```

## Scope and limitations

This is a controlled diagnostic suite, not a population estimate for all
agents or repositories. Natural failures are rare and selected from calibrated
cells. Persistent faults are simulator-state positive controls. The soft and
repair cells are small, and the repair experiment is reported with Wilson
intervals as a diagnostic rather than a ranking. Classification depends on a
finite replay budget `N`; unstable/no-Delta is a substantive outcome, not a
row to discard.

## Verification and CI

GitHub Actions runs the deterministic test suite, paper-claim verifier,
artifact integrity checks, and release checks across Python 3.10–3.12. The
verification path uses the committed evidence and requires no GPU, model
server, external API, or secret.

## Citation

See [`CITATION.cff`](CITATION.cff) and
[`paper/citation.bib`](paper/citation.bib).

The workshop is non-archival. Cite the OpenReview workshop paper, not an ICML
main-conference or PMLR proceedings paper.

## License

Code and the released artifact are available under the
[MIT License](LICENSE).
