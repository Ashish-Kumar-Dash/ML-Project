# MOSI evaluation-split gate — seed 1, aligned MOSI, drop_rate 0.7

Run: 2026-09-30 17:50:20–17:52:26 IST, exit 0, wall-clock 2:06, peak GPU 925 MiB, peak host RSS 1.8 GiB.
Workstation dir: `ML-Project/runs/mosi_gate/` (train.log, gate.patch, apply_gate_patch.py, run.sh, eval_fix_c6f8d18.patch, mosi_seed1.pt sha256 77c694323c0e59b8…).

## Command
```
python -u main.py --pretrained_model pretrained/mosei.pt --dataset mosi --data_path data/mosi_data.pkl \
  --drop_rate 0.7 --seed 1 --name runs/mosi_gate/mosi_seed1.pt      # batch_size default 64, 40 epochs
```
Code: MPLMM c6f8d18 + compat_torch2.14.patch (weights_only=False at utils.py:81 and train.py:159, ReduceLROnPlateau verbose removed) + gate.patch (DataParallel removed at train.py:66/119; end-of-run evaluation block).
Backbone: pretrained/mosei.pt (sha256 6a29d2c3…).

## Design: one run, not two
`test=False` and `test=True` runs are identical until train.py:160, so the gate trains once. After reloading the best checkpoint it saves the RNG state, then restores that state before evaluating each split. That reproduces what each of the two runs would print. As a check, the upstream line-160 print matches the `valid` block exactly.
(gate.patch also re-evaluated the checkpoint under 20 mask seeds. Those outputs were removed as out of scope, and the numbers below do not depend on them.)

## Result
Best checkpoint is from epoch 32 (lowest validation loss 1.0948).

| split | code path | n (non-zero) | Acc-2 | F1 | MAE | Corr | Acc-7 | Acc-5 |
|---|---|---|---|---|---|---|---|---|
| valid | upstream `test=False` | 229 (216) | **68.06** | 67.91 | 1.1531 | 0.558 | 25.76 | 31.00 |
| test | `test=True` | 686 (656) | **69.82** | 69.63 | 1.1475 | 0.506 | 27.70 | 30.76 |

Paper value: **72.14**.

## Outcome: neither matches
Validation is −4.08 points and test −2.32 points away from 72.14. Following the gate's decision rule, this is a reproduction offset: documented here, and the work moves on.
- Single training seed (1).
- 72.14 is taken to be binary accuracy excluding zero labels, which is what `eval_mosi(..., exclude_zero=True)` prints.

## Fixes carried forward (eval_fix_c6f8d18.patch, applies to upstream c6f8d18)
- train.py:160: `evaluate(..., test=True)`.
- train.py:159 and utils.py:81: `torch.load(..., weights_only=False)`.
- train.py:66 and :119: DataParallel removed (single GPU).
- ReduceLROnPlateau `verbose` removed (torch 2.14 compatibility).
- Always pass `--drop_rate 0.7` explicitly; main.py:80 defaults to 0.6.

## Not addressed (outside the blocking issue)
The validation/test missing-modality masks are drawn with Python `random` on every evaluation pass (src/mosidata.py:52-56). This is left for the team to decide on.
