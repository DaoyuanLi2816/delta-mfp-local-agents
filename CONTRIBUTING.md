# Contributing

Contributions that improve the replay implementation, add auditable diagnostic
tasks, or strengthen reproducibility are welcome.

## Development checks

```bash
make setup
make test
make verify-paper
make check-release
```

The default checks require no GPU, Ollama server, model weights, or network
access. Keep model-backed experiments behind explicit commands.

## Result integrity

- Do not overwrite committed result CSVs or retained traces during a routine
  test run.
- If correcting a result, document the affected estimand, rows, command,
  environment, and before/after values.
- Keep prefix-0, nontrivial Delta-MFP, unstable/no-Delta, and irreversible
  outcomes in the denominator. Do not discard unstable replays.
- New repair comparisons must report replay budget, cell size, and uncertainty;
  the existing low-N repair probe is diagnostic, not a ranking.

## Data and safety

Do not commit credentials, model weights, private traces, absolute user paths,
or identifying task data. Run `make check-release` before publishing.
