# SOAP Optimizer Trick

## Method overview

SOAP = **Adam in Shampoo eigenbasis** (https://arxiv.org/abs/2409.11321).
For each matrix parameter, gradients are projected onto the eigenvectors of
a slow-moving Shampoo preconditioner; Adam runs in that rotated basis;
updates are projected back. Eigenbases are refreshed every
`precondition_frequency` steps via QR iteration (initial decomposition via
`torch.linalg.eigh`).

In this trick, SOAP **replaces Muon** for transformer-block matrix parameters.
`tok_emb` / `lm_head` / scalar / skip parameters keep using Adam (same as
baseline) so the comparison isolates the matrix-optimizer change.

## Origin

`class SOAP` in `train_gpt.py` is **inlined verbatim** from
https://github.com/nikhilvyas/modded-nanogpt-SOAP/blob/master/soap.py
(no algorithm changes). Only the integration glue (env-var hyperparam
reads, optimizer instantiation, W&B logging) was written for this project.

Convention mirrors `Bag-of-tricks-for-Transformers/exp/muon-ema-update-smoothing/`:
inline optimizer class, no `ENV_KEY_MAP` changes, no auxiliary `.py` files.

## Key differences vs. baseline-sp1024

| Aspect                    | Baseline               | SOAP                |
| ------------------------- | ---------------------- | ------------------- |
| matrix optimizer          | Muon                   | SOAP                |
| `soap_lr`                 | —                      | 3e-3                |
| betas                     | muon_momentum=0.95     | (0.95, 0.95)        |
| `precondition_frequency`  | —                      | 10                  |
| `weight_decay`            | —                      | 0.01                |
| Muon momentum warmup      | 0.85→0.95 over 500 stp | removed             |

Everything else (model, data, tokenizer, schedule, eval) unchanged.

## Impact on training

- **Memory**: per 2D matrix param, adds fp32 `GG` (d×d) and `Q` (d×d)
  preconditioner buffers. For the 9-layer × dim=512 model, expected
  ~+70-100 MiB over baseline.
- **Compute**: per-step Adam path + one `eigh`/`QR` per
  `precondition_frequency=10` steps. Expected step_avg_ms increase <5%
  on 8×H100.
- **Convergence**: reference implementation reports val_loss
  3.271 → 3.256 on NanoGPT 124M (10B tokens). Our model is smaller; result TBD.

## BPB analysis (filled after run)

| Regime           | Baseline BPB | SOAP BPB | Δ BPB |
|------------------|--------------|----------|-------|
| fixed_compute    | TBD          | TBD      | TBD   |
| fixed_tokens_10b | TBD          | TBD      | TBD   |

## Overrides

The harness's `ENV_KEY_MAP` does **not** forward `SOAP_*` keys (project
convention — see `muon-ema-update-smoothing` for the same pattern).
Override defaults via shell env:

```bash
SOAP_LR=1e-3 SOAP_PRECONDITION_FREQUENCY=20 \
  uv run python exp/run_experiments.py exp/soap/soap.json
```

If OOM: `SOAP_MERGE_DIMS=1` or lower `SOAP_MAX_PRECOND_DIM`.

## Files

- `train_gpt.py`: SOAP-enabled trainer (cp from `baseline-sp1024/train_gpt.py`,
  with `class SOAP` inlined and Muon swapped out at instantiation site).
- `soap.json`: experiment manifest (two regimes: fixed_time_10min, fixed_tokens_10b).
- `run.sh`: convenience launcher.
- `logs/`: per-batch outputs (auto-generated).
