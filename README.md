# Length-Invariant Missing-Modality Generation

Diagnosing and repairing the generation module of [MPLMM](https://aclanthology.org/2024.acl-long.94) (Guo, Jin & Zhao, ACL 2024).

## The finding

MPLMM handles absent modalities by freezing a MulT backbone and learning three prompt types plus a small generation module. The paper's own Limitations section predicts the module will fail on raw features, blaming weaker inter-modal correlation, and leaves it as future work.

Reading the released code gives a different explanation. The module's time-mixing layer is a `Conv1d(kernel_size=1)` applied *after a transpose*, which puts time in the convolution's channel slot. The kernel restricts the feature axis, not time — so the layer is a dense, position-indexed map over time: every output position draws on every input position, with no weight sharing and no translation-equivariance. Its parameter count is welded to the exact sequence length it was built at.

There are nine such layers, and the count is bilinear in output length and total input length:

```
params = ℓ_out × (ℓ_p + Σ ℓ_in) + ℓ_out
```

| CMU-MOSI | 9 time-mixing layers | all length-dependent params | trainable (vs 1,071,211-param backbone) |
|---|---|---|---|
| aligned (50/50/50)     |  37,650 |    55,650 |    88,141 — 8.2%   |
| unaligned (50/375/500) | 972,175 | 1,083,175 | 1,115,666 — **104%** |

The paper states its trainable parameter count does not grow with backbone size. True, and not the binding constraint: it grows with *input sequence length*, and at unaligned lengths the length-dependent parameters exceed the backbone they exist to avoid fine-tuning.

Reproduce: `python tools/count_params.py`

## Questions

1. **Structural** — how do the module's parameter count and inductive bias vary with sequence length, and what failure mode does that predict?
2. **Diagnostic** — when representations become less affect-aligned, is the operative variable correlation (as claimed), dimensionality, or length?
3. **Constructive** — can the layer be replaced by a length-invariant operator under the original parameter budget?

Q1 is answered analytically. Q2 by a controlled ladder in which each rung moves one factor. Q3 ships regardless of what Q2 returns.

## Status

- [x] Parameter audit — `tools/count_params.py`
- [ ] L1 baseline reproduction (target: within 2 pts of 72.14 avg ACC, or a documented offset)
- [ ] `l_avp` learned-weight heatmap + receptive-field probe
- [ ] Length-invariant operator vs dense `l_avp`, fixed data, unaligned MOSI
- [ ] Pooled length ladder, ℓ ∈ {50, 100, 200, 375/500}, with measured CKA
- [ ] wav2vec2 / CLIP rungs

## Layout

```
src/        instrumented fork of zrguo/MPLMM
tools/      parameter audit, CKA, probes
results/    run outputs, checked in
docs/       CLAIMS.md · SETUP.md · DEVIATIONS.md
figures/
```

`docs/CLAIMS.md` maps every architectural claim above to a `file:line` in upstream and how it was verified.
`docs/DEVIATIONS.md` records where we depart from the proposal.

## Setup

See `docs/SETUP.md`. Two things that will bite immediately:

- `transfer_model` calls `torch.load` on a pickled `MULTModel`; needs `weights_only=False` on PyTorch ≥ 2.6.
- Pass `--drop_rate 0.7` explicitly. The paper's reported optimum is 0.7; the code defaults to 0.6.

Feature pickles: use the MulT release (GloVe-300 text). MMSA's `Processed/*.pkl` ship BERT text features and are **not** interchangeable.

## Team

Tanmay Jaiswal — feature pipeline and data engineering
Sujal Som — model and training
Ashish Kumar Dash — evaluation and analysis

ML course project, IIT Bhilai. Backbone: [MulT](https://github.com/yaohungt/Multimodal-Transformer). Upstream: [zrguo/MPLMM](https://github.com/zrguo/MPLMM).
