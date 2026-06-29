# Predicted Repair Router Rules

The predicted routing baseline in this paper is **rule-based, not learned**. It
inspects only prefix-visible features in the recorded trajectory and selects a
single repair gate. The rules are designed to be inspectable so reviewers can
see exactly what signals drive predicted routing without retraining a model.

## Features

For each trace we serialize the entire trajectory (action records, tool
results, memory writes, parse statuses) and check for the following markers:

| Feature | Marker(s) in trace |
| --- | --- |
| `untrusted_external` | substrings `untrusted`, `credentials`, `external_source` |
| `id_mismatch` | substrings `wrong-`, `wrong_id_hint`, `distractor_id` |
| `missing_field` | substrings `field_removed`, `"missing"`, `missing required` |
| `ambiguous_date` | substrings `tomorrow`, `ambiguous_date_hint`, `date_ambiguous` |
| `precondition_skip_marker` | substring `skip_required_check` |
| `stale_memory` | substrings `stale_lookup`, `stale_summary` |
| `mutating_before_check` | scan: a mutating tool action precedes any check tool action |
| `parse_failure` | substring `parse_failure` |

The first six features are observable cues (string signals embedded in tool
output, memory writes, or arguments). The seventh is a structural cue derived
by walking the action sequence. The eighth is the parser status produced by
the interface adapter.

## Rule order

Rules fire in order; the first matching rule wins.

```
1. untrusted_external          -> BoundaryShield
2. id_mismatch                 -> EvidenceGate
3. missing_field               -> ClarifyGate
4. ambiguous_date              -> ClarifyGate
5. precondition_skip_marker    -> PreconditionGate
6. stale_memory                -> RollbackRetry
7. mutating_before_check       -> PreconditionGate
8. parse_failure               -> ParseRepair
9. (default)                   -> RollbackRetry
```

Specific cues are checked before generic ones. Mutating-before-check is a
conservative fallback because it can fire on many normal traces where the
agent legitimately mutates state after a clarification. Putting it after the
specific markers keeps it from masking better-matched repairs.

## Why this is not a trained model

We deliberately do not train a router. With single-replay budgets and 40
persistent + ~15 soft fault traces, a learned router would overfit. The
rule-based router is interpretable and auditable; a single bug or marker
mismatch shows up directly in the routing distribution.

## Routing distribution on the persistent fault set (n=40)

| Fault type     | n  | Predicted route   |
|----------------|----|-------------------|
| wrong_id       | 10 | EvidenceGate      |
| stale_memory   | 10 | RollbackRetry     |
| precondition_skip | 10 | PreconditionGate  |
| date_ambiguity | 7  | ClarifyGate       |
| missing_field  | 3  | ClarifyGate       |

This matches the oracle map for every persistent fault subtype after
correcting the router's window from "last 4 steps" to "all steps", because
fault markers can surface at the perturbation prefix or at any later step
where the agent first acts on the bad evidence. On soft faults, predicted
routing is expected to be noisier because perturbations are quieter.

## What the router does *not* claim

- It does not promise to recover natural failures with no marker tokens.
- It does not differentiate fault classes that produce the same surface marker
  (e.g. soft and persistent wrong-ID look identical in the trace).
- It is a strong starting point rather than a learned decision rule.
