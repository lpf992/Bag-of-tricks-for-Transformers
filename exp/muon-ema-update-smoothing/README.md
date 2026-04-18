# Muon EMA Update Smoothing

## Method

This experiment adds **EMA (Exponential Moving Average) update smoothing** to
the Muon optimizer, adapted from modded-nanogpt PR #129 (Ashok Cutkosky).

After computing the standard Muon update (Newton-Schulz orthogonalization of
the Nesterov-momentum gradient), a second EMA is applied to the sequence of
updates:

    smoothed_buffer = s * smoothed_buffer + (1 - s) * update
    final_update    = s * smoothed_buffer + (1 - s) * update

where `s` is the smoothing weight. The second line gives a Nesterov-like
"look-ahead" on the smoothed trajectory.

The smoothing weight decays linearly from 0.5 to 0.2 over the first 3000
steps, matching the original source's schedule.

## Key Differences from Baseline

| Parameter | baseline-sp1024 | muon-ema-update-smoothing |
|---|---|---|
| Muon `update_smoothing` | 0.0 (none) | **0.5 -> 0.2 over 3000 steps** |
| `smoothed_update_buffer` | N/A | **bfloat16 buffer per matrix param** |
| All other hyperparameters | (unchanged) | (unchanged) |

## Origin

- Source: [modded-nanogpt PR #129](https://github.com/KellerJordan/modded-nanogpt/pull/129)
- Author: Ashok Cutkosky
- Commit: `16a6690`
- Original result: ~1% speedup on medium track (23.52 min vs 23.72 min, p=0.000032 over 160 runs)

## Impact on Training

- **Memory**: One additional bfloat16 buffer per Muon-managed parameter
  (matrix weights in transformer blocks). Modest compared to existing fp32
  momentum buffers.
- **Compute**: Two extra elementwise operations per Muon parameter per step
  (multiply-add for EMA, multiply-add for Nesterov blend). Negligible
  compared to the Newton-Schulz iteration.
- **Convergence**: Expected to smooth the optimization trajectory, reducing
  oscillation in later training and potentially finding flatter minima.

## Results

### Fixed Compute (10 min wall-clock)

| Metric | baseline-sp1024 | muon-ema-update-smoothing | Delta |
|---|---|---|---|
| **Val BPB** | 1.2194 | TBD | TBD |
| Val Loss | 2.0589 | TBD | TBD |

### Fixed Tokens (10B tokens)

| Metric | baseline-sp1024 | muon-ema-update-smoothing | Delta |
|---|---|---|---|
| **Val BPB** | 1.2118 | TBD | TBD |
| Val Loss | 2.0460 | TBD | TBD |

## Analysis

TBD after experiments complete.

## Files

- `train_gpt.py`: Training script with EMA update smoothing
- `muon-ema-update-smoothing.json`: Experiment manifest
- `logs/`: Training output (automatically generated)
