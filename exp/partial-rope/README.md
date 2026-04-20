# Partial RoPE — rotate only the leading `rope_dims` of each head

## Method

Baseline attention applies Rotary Position Embedding (RoPE) to the **full**
head dimension `D = head_dim = 64`. Partial RoPE restricts the rotation to the
first `rope_dims` entries of each head and lets the remaining
`D − rope_dims` dimensions pass through unchanged:

```
x            = [ x_rope        | x_pass        ]          shape: [..., D]
               len = rope_dims   len = D − rope_dims

x_rope_out   = rotate_split_half(x_rope, cos, sin)        (same scheme as baseline)
output       = concat(x_rope_out, x_pass, dim = -1)
```

The cos/sin cache inside `Rotary` is built from `rope_dims` instead of
`head_dim`, so when `rope_dims > 0` and `rope_dims < head_dim` the cache has
size `rope_dims / 2` and `apply_rotary_emb` auto-detects the partial mode via
`rd = cos.size(-1) * 2`.

Setting `rope_dims = 0` (the default) restores the full-RoPE baseline
bit-exactly — `Rotary` caches `head_dim / 2` cos/sin entries and
`apply_rotary_emb` takes the else branch, identical to `exp/baseline-sp1024`.

**Constraint.** `rope_dims` must be even and `≤ head_dim`.

**Intuition.** The rotated dims carry positional information (relative
distance enters the attention score via `q·k`); the pass-through dims behave
like a position-invariant embedding — free to learn content-only features.
At `rope_dims = 16 / 64 = 25 %` the reference paper reports materially
better `val_bpb` than full RoPE at zero parameter cost.

**Distributed correctness.** Pure forward-pass change on per-token tensors;
no new state, no collectives, no RNG draws. DDP behavior is identical to
baseline.

## Single-axis experiment (the 2 runs)

Both runs use `rope_dims = 16` (25 % of head_dim = 64, matching the
parameter-golf record) and differ only in the control regime:

| Experiment | `rope_dims` | Control |
|---|---|---|
| `partial-rope-16-fixed_time_10min` | 16 (25 % of 64) | fixed_compute 600 s |
| `partial-rope-16-fixed_tokens_10b` | 16 (25 % of 64) | fixed_tokens 10 B  |

## Key differences from baseline

| Parameter | baseline-sp1024 | partial-rope |
|---|---|---|
| `rope_dims` | N/A (rotates full 64 dims) | 16 (rotates only first 16 of 64) |
| Rotary cos/sin cache size | `head_dim / 2 = 32` | `rope_dims / 2 = 8` |
| `apply_rotary_emb` | split-half on full tensor | split-half on first 16 dims, concat rest |
| Parameters | (unchanged) | **+0** |
| Memory | (unchanged) | **unchanged** (smaller cos/sin cache is negligible) |
| Compute | (unchanged) | slightly **less** (~75 % fewer rotation mul/adds) |
| All other hyperparameters | (identical) | (identical) |

## Origin

- Source record:
  [openai/parameter-golf — `2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248`](https://github.com/openai/parameter-golf/blob/main/records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md)
- Submission BPB: `1.1248` (vs. `1.1271` under the preceding record), at
  `rope_dims = 16 / 64 = 25 %`.
- Ablation note from the record: the companion "Late QAT" flag in the same
  submission was later shown to be dead code (folded by `torch.compile`), so
  the full improvement over `1.1271` is attributed to Partial RoPE + LN Scale.
  This project replicates Partial RoPE alone; LN Scale is out of scope here.

## Impact on training

- **Memory.** `Rotary._cos_cached` / `_sin_cached` shrink from `[1, 1, T, 32]`
  to `[1, 1, T, 8]` — negligible in absolute terms.
- **Compute.** Each attention layer does one `apply_rotary_emb` on Q and one
  on K; the rotated segment shrinks 4×, so RoPE mul/add cost drops ~75 %.
  In practice this is already a small fraction of attention FLOPs, so wall-
  clock speed-up is well under 1 %.
- **Convergence.** The 48 pass-through dims remove positional bias from a
  majority of the head space, letting those dims carry content-only features
  that can be attended to position-invariantly. Expected to lower `val_bpb`
  at identical parameter count and wall clock.

## Results

### Fixed Compute (10 min wall-clock)

| Metric | baseline-sp1024 | partial-rope-16 |
|---|---|---|
| **Val BPB** | 1.2194 | TBD |
| Val Loss   | 2.0589 | TBD |

### Fixed Tokens (10 B tokens)

| Metric | baseline-sp1024 | partial-rope-16 |
|---|---|---|
| **Val BPB** | 1.2118 | TBD |
| Val Loss   | 2.0460 | TBD |

## Analysis

TBD after experiments complete. If 25 % looks good we can follow up with a
ratio sweep (e.g. `rope_dims ∈ {8, 32, 48}`) to locate the sweet spot for
this architecture (head_dim = 64, 9 layers, 512 model_dim).

## Files

- `train_gpt.py` — trainer with Partial RoPE wiring (look for `# trick: partial-rope`)
- `partial-rope.json` — 2-experiment manifest
- `run.sh` — launch both experiments and sync offline wandb runs
- `logs/` — experiment outputs (automatically generated)

## How to run

```bash
# Dry-run the manifest (prints 2 resolved runs without launching)
python exp/run_experiments.py exp/partial-rope/partial-rope.json --dry-run

# Launch both experiments
bash exp/partial-rope/run.sh
```
