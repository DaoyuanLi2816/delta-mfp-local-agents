# Released results

Run `python scripts/verify_paper_claims.py --json` for the machine-readable
expected-versus-actual report.

## Failure regimes

| Evidence block | Nontrivial Delta-MFP | Prefix-0 | Unstable / no-Delta |
|---|---:|---:|---:|
| Natural failures (`N=3`, 25 traces) | 13 | 5 | 7 |
| Persistent positive control (`N=2`, 40 traces) | 40 | 0 | 0 |
| Soft perturbations (`N=2`, 50 traces) | 7 | 20 | 23 |
| Soft perturbations (`N=5`, 50 traces) | 7 | 22 | 21 |
| Qwen2.5-14B soft probe (`N=3`, 24 traces) | 5 | 0 | 19 |

All 40 persistent cases have zero distance between the injected prefix and the
localized Delta-MFP. This validates the controlled replay path; it is not an
estimate of natural-failure localization accuracy.

## Replay-budget stability

The aggregate soft profile changes only slightly from `7/20/23` at `N=2` to
`7/22/21` at `N=5`, but only one of the seven `N=2` nontrivial traces remains
nontrivial. Six other traces switch into the nontrivial category, and only
37/50 traces retain their regime. The stable aggregate count therefore hides
substantial per-trace turnover.

## Repair probe

The repair cells are deliberately small. Persistent no-repair succeeds
`0.000`, while restart-style Retry succeeds `1.000` on the positive control.
On soft traces, the no-repair baseline is already `0.667`; the fault-label
Oracle is `0.083`. These estimates have wide Wilson intervals and should not
be used to rank repair methods.
