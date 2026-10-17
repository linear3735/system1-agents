# Intern-Decision accuracy-v1

Verify pinned benchmark inputs and replay saved responses with the official
Intern evaluator. Run from the repository root with the public231 runner's shared
`benchmarks.client`, `benchmarks.evidence` and `benchmarks.legs.leg1_public231` modules.
`verify` uses only the standard library; replay needs the upstream evaluation
packages, including Torch, in a prepared CPU environment. No datasets or weights
are downloaded by this entry.

## Sources

| Repository | Required clean revision |
|---|---|
| [Intern-Decision](https://github.com/InternLM/Intern-Decision) | `3572c8a68b5df5dafe02d0e093989ba8ec0183bc` |
| [JevBench](https://github.com/fstandhartinger/jevbench) | `7ce310c7262ed49cc85853339a8a42459298e3f3` |

The upstream bundle verifier checks hashes and the complete plan:

| Suite | Requests | Decisions |
|---|---:|---:|
| JevBench easy / original / hard | 48 / 72 / 111 | 48 / 72 / 111 |
| AG News | 7,600 | 7,600 |
| ToolACE | 310 | 310 |
| Typed Decisions | 400 | 2,000 |
| WildJailBreak | 2,210 | 2,210 |
| Total | 10,751 | 12,351 |

Public JevBench is a development diagnostic, not the full leaderboard or an
independent holdout. The manifest lists AG News licensing as `unknown`; this
entry preserves that metadata and does not redistribute the inputs.

## Verify and replay

```bash
python -m benchmarks.intern_decision verify \
  --intern-root /path/to/Intern-Decision --scorer-root /path/to/jevbench
python -m benchmarks.intern_decision replay \
  --intern-root /path/to/Intern-Decision --scorer-root /path/to/jevbench \
  --round /path/to/saved-run/round-1 --out /path/to/fresh-replay
```

Keep outputs outside the reference checkouts and use a new output directory for
each invocation. `JEVBENCH_ROOT`, if set, must match `--scorer-root`.

Replay accepts public231 `round-1` through `round-3`, or saved seven-suite
accuracy-v1 evidence. It verifies `environment.json` against its recorded hash and
every original environment field in `run.meta.json`, plus the pinned data hashes.
Every `results.jsonl` attempt must identify `(suite, physical source_line, task_id)`
and match its raw file hash. Duplicate or unknown identities, missing raw files
and raw files without matching ledger entries are rejected.
The entire saved request must match the frozen task, original field/option order,
`thinking.enabled=false` and the configured model; extra fields are rejected.
`response_body` is authoritative. Decoded answers remain unchanged; HTTP, client,
parse and refusal failures remain failed attempts. Scored ledger columns are ignored.

`input.json` records metadata, ledger and raw paths/hashes; `report.json` binds it
by hash. Retain the original run because raw files are referenced, not copied.
Recorded configuration is provenance, not proof of the loaded weights.

## Scoring and completeness

The unmodified upstream `src.eval.jev.evaluate` and table renderer perform scoring.
Answers need per-field `probabilities`; scalar-only responses cannot complete this
replay. No replacement distributions or temperature calibration are applied.
HF T=1 and XTuner calibrated results describe different inference settings.

`report.json` always keeps all seven planned denominators. Missing responses or
an official evaluation failure leave that suite incomplete with an error and
unknown scores. Numeric overflow is a suite failure, never a clipped probability.
Only seven successful suite evaluations produce `complete.json`, `accuracy.md`
and `average_accuracy`, the unweighted mean of seven accuracies. Replaying only
the three public suites cannot produce a complete seven-suite result.
`replay` exits 0 after writing an incomplete report, so callers must check
`report.json.complete` and each suite's error; rejected evidence exits nonzero.

The pinned Intern Hard input has no `gold_probs`: official `tvd_n=0`, `tvd=null`
means TVD was not evaluated. Original JevBench Hard has 10 references, but is a
different input. This entry calls official `evaluate` directly because upstream's
main/merge CLI requires those 10 references even for the bundled input. The table's
Brier/ECE columns refer to JevBench hard. A complete replay establishes scoring
coverage, not reproduction of published model quality or inference latency.
