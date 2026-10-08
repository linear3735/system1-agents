# Intern-Decision accuracy-v1 on one H800

Three pinned Intern checkpoints used the official HF service, BF16, SDPA and
temperature 1. Each received five warmups, then one serial pass over all seven
suites: 10,751 requests and 12,351 decisions. All responses were valid.

## Results

| Suite | Requests / decisions | Intern 0.8B | Intern 2B | Intern 4B |
|---|---:|---:|---:|---:|
| JevBench easy | 48 / 48 | 47/48 | 48/48 | 48/48 |
| JevBench original | 72 / 72 | 57/72 | 61/72 | 71/72 |
| JevBench hard | 111 / 111 | 60/111 | 72/111 | 81/111 |
| AG News | 7,600 / 7,600 | 6,735/7,600 | 6,833/7,600 | 6,903/7,600 |
| ToolACE | 310 / 310 | 293/310 | 299/310 | 299/310 |
| Typed Decisions | 400 / 2,000 | 1,552/2,000 | 1,582/2,000 | 1,613/2,000 |
| WildJailBreak | 2,210 / 2,210 | 1,414/2,210 | 1,731/2,210 | 1,982/2,210 |
| **Mean of seven accuracies** | Each suite has equal weight | **79.4077%** | **84.7675%** | **89.8854%** |

The mean uses each suite's unrounded accuracy, not the pooled fraction over
12,351 decisions. Typed Decisions keeps all five fields in one request.
These are HF T=1 results; no XTuner calibration or published-score equivalence
is claimed. The pinned Hard input has no `gold_probs`, so TVD is unmeasured.

## Reproduce

Use the [verify, collect and replay commands](../INTERN_DECISION.md). Inputs and
the official evaluator are pinned to Intern-Decision
`3572c8a68b5df5dafe02d0e093989ba8ec0183bc`; JevBench is
`7ce310c7262ed49cc85853339a8a42459298e3f3`.

| Checkpoint | Revision |
|---|---|
| Intern-Decision-0.8B | `85a0cc5a99d67ea8d56dfe98115689212867171d` |
| Intern-Decision-2B | `8797836c65fc91a2435b1fb6850b5f0aabd75cc3` |
| Intern-Decision-4B | `0e5e6aa7d6d750e2b1504ba11a8136cb58aeb3cd` |

The service used PyTorch 2.9.1+cu128, Transformers 5.14.1, an 8192-token limit,
thinking off and no KV cache. Optional FLA/causal-conv1d paths were unavailable.
This run was collected on 2026-10-07 UTC, Slurm job 415106. It is a separate
allocation from the [five-model public231 comparison](public231-h800.md);
latencies from those two runs should not be combined.

A non-author checked every saved request and response against the fixed inputs,
then independently re-scored them and called the unmodified official `evaluate`
on CPU. Counts and metrics agreed. Source, environment and response hashes are
retained with the run. The data bundle is not vendored; its manifest leaves
AG News licensing unresolved. The public attachment retains predictions, scores
and timing, omitting input states and returned legends. Full raw evidence stays
with the original run; the public copy is not a complete replay bundle.
