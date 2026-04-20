# Snoo — Sparse Nesterov Outer Optimizer

## Method

Snoo is a **look-ahead wrapper** around an existing inner optimizer. It does
not replace the inner step; it runs alongside it and, every `k` inner steps,
applies a Nesterov-SGD update to the accumulated parameter displacement.

Let `p` be a wrapped parameter, `p_old` a slow copy (the outer buffer), and
`Δ = p − p_old` the k-step drift produced by the inner optimizer. Every `k`
steps Snoo performs:

    p.grad   = p_old - p           # pseudo-grad: −Δ
    p        ← p_old               # rewind
    SGD(lr, momentum, nesterov).step()   # consumes the pseudo-grad
    p_old    ← p                   # commit new slow copy

All other steps are no-ops. The net effect is a temporally smoothed trajectory
biased towards flatter minima at the cost of one extra parameter-sized buffer
and one SGD step every `k` inner steps.

**Distributed correctness.** The pseudo-gradient `p_old − p` is computed
locally on each rank. Because `p` is kept in lock-step across ranks by the
inner optimizer (DDP backward all-reduce + Muon's internal all-reduce), and
`p_old` is initialized identically and updated in parallel, the pseudo-grad is
identical on every rank — no extra collective is required inside Snoo.

## Scope ablation (the 4 experiments)

To isolate which inner optimizer benefits from the outer smoothing, Snoo is
applied to a **subset** of parameters rather than the whole model (PR#128
wraps every parameter). Two scopes × two control regimes = 4 runs:

| Experiment | Scope wrapped by Snoo | Control |
|---|---|---|
| `snoo-on-muon-fixed_time_10min`  | `matrix_params` (Muon-governed)            | fixed_compute 600 s |
| `snoo-on-muon-fixed_tokens_10b`  | `matrix_params`                            | fixed_tokens  10 B  |
| `snoo-on-adamw-fixed_time_10min` | `tok_emb + lm_head + scalar_params`        | fixed_compute 600 s |
| `snoo-on-adamw-fixed_tokens_10b` | `tok_emb + lm_head + scalar_params`        | fixed_tokens  10 B  |

All four use the PR#128 defaults: `lr = 0.68`, `momentum = 0.37`, `k = 28`.

## Key differences from baseline

| Parameter | baseline-sp1024 | snoo |
|---|---|---|
| Outer optimizer | none | SGD(lr=0.68, momentum=0.37, nesterov=True) every 28 inner steps |
| Outer buffer | none | one `torch.clone()` per wrapped parameter (≈ +100 % memory for those params only) |
| `snoo_scope` | N/A | `muon` or `adamw` (chooses which inner optimizer's params are wrapped) |
| Inner optimizers | unchanged | unchanged |
| All other hyperparameters | (identical) | (identical) |

## Origin

- Source: [modded-nanogpt PR #128](https://github.com/KellerJordan/modded-nanogpt/pull/128)
- Authors: @dominikkallusky, @vishal9-team, @vinaysrao
- Merge commit: `7aae096` (2025-12-31)
- Original result: 60-step / ~10 s improvement on the medium track

## Impact on training

- **Memory.** One extra clone per wrapped parameter. For `snoo-on-muon`
  that's roughly the matrix-parameter footprint (the dominant share of
  the model); for `snoo-on-adamw` it's the embedding + head + scalars.
- **Compute.** Every 28th step, one SGD-Nesterov step over the wrapped
  parameters plus two copy passes. Amortizes to well under 1 % of a
  training step; no Newton-Schulz or extra backward is triggered.
- **Convergence.** Expected to reduce oscillation in the wrapped params
  and bias towards flatter minima. The scope split lets us see whether
  the benefit localizes to matrix updates or embedding/head updates.

## Results

### Fixed Compute (10 min wall-clock)

| Metric | baseline-sp1024 | snoo-on-muon | snoo-on-adamw |
|---|---|---|---|
| **Val BPB** | 1.2194 | TBD | TBD |
| Val Loss   | 2.0589 | TBD | TBD |

### Fixed Tokens (10 B tokens)

| Metric | baseline-sp1024 | snoo-on-muon | snoo-on-adamw |
|---|---|---|---|
| **Val BPB** | 1.2118 | TBD | TBD |
| Val Loss   | 2.0460 | TBD | TBD |

## Analysis

TBD after experiments complete.

## Files

- `train_gpt.py` — trainer with Snoo wiring (look for `# trick: snoo`)
- `snoo.json` — 4-experiment manifest
- `run.sh` — launch all 4 experiments and sync offline wandb runs
- `logs/` — experiment outputs (automatically generated)

## How to run

```bash
# Dry-run the manifest (prints 4 resolved runs without launching)
python exp/run_experiments.py exp/snoo/snoo.json --dry-run

# Launch all 4 experiments
bash exp/snoo/run.sh
```
