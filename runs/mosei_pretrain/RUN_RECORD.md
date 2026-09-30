# Run record — MOSEI aligned pre-training

Owner: Tanmay Jaiswal · Host: 10.50.49.43 (RTX 3060 12 GiB, shared workstation) · 2026-09-29

## Output
- Checkpoint: `/home/user/Downloads/ML-project/ML-Project/pretrained/mosei.pt`
- SHA-256 `6a29d2c399d977a91624a0d9f68bda7571c3f824bdbbf2c3ff794c37ef841de6`, 4,614,123 bytes, full pickled `src.model.MULTModel`
- Best validation epoch 39 of 40 (valid L1 0.6094). The checkpoint was saved 15 times (once per validation improvement); the last save was at 23:47:08 IST.
- 1,073,731 parameters, all `requires_grad=True`. Checked on the epoch-1 checkpoint; the flags are set at construction, and no `--pretrained_model` was passed, so `transfer_model` never freezes anything.

## Configuration
`python -u main.py --dataset mosei --data_path data/mosei_senti_data.pkl --drop_rate 0 --batch_size 64 --name pretrained/mosei.pt`

- MPLMM `c6f8d18e9222bd65165a3214820201dc51a35f19`, cloned to `runs/mosei_pretrain/MPLMM`. Not in `src/`, which belongs to Sujal.
- Upstream defaults: 40 epochs, Adam lr 1e-3, ReduceLROnPlateau (patience 10, factor 0.1), clip 0.8, seed 666, 5 layers, proj_dim 30, 5 heads.
- Data: MulT `mosei_senti_data.pkl` (see `docs/PROVENANCE.md`).
- Environment: torch 2.14.0+cu130, torchvision 0.29.0+cu130, Python 3.12.3, driver 580.173.02.

## Wall-clock — the ladder price
| Quantity | Value |
|---|---|
| Process wall-clock (`/usr/bin/time`) | 16:30.24 (990 s) |
| Sum of 40 epoch times | 974.9 s |
| Per epoch (train 16,265 + valid 1,869 + test 4,643) | 24.37 ± 0.19 s (range 24.06–25.32) |
| Fixed overhead (pickle load ×3, model build, final eval) | 15 s |
| Training step | ~79 ms / batch of 64, 254 batches / epoch |
| Peak GPU memory (sampled every 30 s) | 1,535 MiB |
| Peak host RSS | 6.36 GiB |

**Scope.** This measures T = 50 on all three modalities. It does not extrapolate to the longer ladder rungs (ℓ = 100, 200, 375/500), because cost there depends on the length-dependent layers and on attention over longer sequences. Time one epoch on each rung before budgeting.

Peak host RSS is high because `MOSIData` unpickles the full 3.7 GB file separately for each split. On this 15 GiB shared box, don't run two such jobs at once.

## Reported metrics — validation, not test
At the end of training, upstream reloads the best checkpoint and calls `evaluate(model, criterion, test=False)`, so these numbers are **validation** scores. The printed MAE equals the best valid loss.

MAE 0.6094 · Corr 0.6139 · Acc-7 0.5040 · Acc-5 0.5152 · Acc-2 0.7981 · F1 0.8017

The per-epoch test L1 at the selected epoch (39) was 0.6383. Don't quote any of these as test results without a separate test-set evaluation.

## Deviations from upstream (compatibility only)
`runs/mosei_pretrain/compat_torch2.14.patch`, 3 lines:
1. `src/train.py`: removed `verbose=True` from `ReduceLROnPlateau` (TypeError on torch 2.14). It only affected printing.
2. `src/train.py`, `src/utils.py`: `torch.load(..., weights_only=False)`. Since torch 2.6 the default refuses full-module pickles, which breaks the final reload and fine-tuning's `transfer_model`.
3. Added `torchvision`, which `src/iemodata.py` imports at module level even for MOSEI runs. It was not in the SOP package list.

The first launch failed at import (missing torchvision) before any training; its logs are in `runs/mosei_pretrain/failed_attempt1/`.

## Files
`runs/mosei_pretrain/`: `manifest.txt`, `run.sh`, `train.log`, `walltime.txt`, `time_v.txt`, `gpu.csv`, `compat_torch2.14.patch`
