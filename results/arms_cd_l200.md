# Arms C and D at ℓ=200 — the pooling ablation

**Protocol.** Six-case forced mean Acc-2: each of cases 0–5 forced across all 686 CMU-MOSI
test samples, averaged. Paper-comparable; excludes complete data. Binomial SE at n=686 is
1.91 points, so single-seed differences under ~2 points are not interpretable.

**Grid.** ℓ=200 rung (`data/ladder/mosi_data_noalign_l200.pkl`), 200 epochs, `when=30`,
bs 16 × accum 4, lr 1e-3, `drop_rate` 0.7, seeds 1/2/3, one machine (RTX 3060,
10.50.21.144), one launcher (`runs_e200/launch_e200.py`). n=3 per arm, 12 runs total.
Every number below is greppable in `runs_e200/mosi_ladder_l200_<arm>_seed<n>/results.json`.

## Results

| arm | operator | trainable | six-case mean ± SD | complete data (case 6) | text present | text absent | CKA fidelity |
|---|---|---:|---:|---:|---:|---:|---:|
| **A** dense `l_avp` (the paper) | `mlp_dense` | 349,441 | 66.65 ± 2.07 | 77.79 | 77.85 | 55.45 | 0.273 |
| **B** length-invariant operator | `cross_attn` | 133,471 | 66.85 ± 1.32 | 78.51 | 78.46 | 55.25 | 0.154 |
| **C** dense, frozen at random init | `mlp_dense_frozen_at_init` | 86,491 | 65.15 ± 0.52 | 78.40 | 77.90 | 52.41 | 0.267 |
| **D** fixed average pool, zero parameters | `avg_pool_fixed_zero_param` | 86,491 | 66.98 ± 0.79 | 77.79 | 78.10 | 55.86 | 0.279 |

Arms C and D train an **identical** parameter set: 349,441 − 262,950 = 86,491 exactly.
C keeps the 262,950 dense time-mixing parameters but never gives them a gradient;
D deletes them. The difference between the two arms is therefore the presence of a
random dense time-mixing map, and nothing else.

## Significance (Welch, n=3 per arm)

| comparison | difference | SE | t | p | 90% CI |
|---|---:|---:|---:|---:|---:|
| B − A | +0.20 | 1.42 | +0.14 | 0.894 | [-2.98, +3.39] |
| C − A | -1.50 | 1.23 | -1.22 | 0.336 | [-4.84, +1.85] |
| D − A | +0.33 | 1.28 | +0.26 | 0.816 | [-2.90, +3.56] |

One-way ANOVA across all four arms: F = 1.25, p = 0.354.

**No arm differs detectably from the paper's dense operator.** Arm D — zero parameters
for time mixing — is +0.33 points against arm A.
This is the claim the ablation was run to test: the dense parameterisation is not merely
inefficient, it is unnecessary at this length.

**It is "no detectable difference", not equivalence.** At n=3 the 90% CI for D − A spans
roughly ±3 points, wider than the ±2 the SOP calls meaningful. n=3 cannot establish
equivalence; it can only fail to find a difference. Say the former.

## Two secondary observations

**A frozen random dense map is worse than no map at all.** D − C = +1.83
points (t = 3.37, p = 0.035), and arm C is the only arm nominally below A. The deficit is
localised: arm C matches the other arms on text-present cases (77.90
vs 77.85 for A) and loses on text-absent cases
(52.41 vs 55.45), i.e.
exactly where the module must synthesise text — consistent with F5. **Do not claim this as
established:** it is one of four pairwise comparisons, the omnibus ANOVA is p = 0.354,
and Holm correction across the pairwise family leaves it non-significant.

**CKA fidelity does not require parameters.** Arm D reconstructs held-out representations as
faithfully as the trained dense operator (linear CKA 0.279 vs
0.273 for A), while the invariant operator sits lower
(0.154). A fixed average pool with no learned parameters matches
the dense module's reconstruction score. This strengthens F6's unplanned finding — fidelity is
not buying task performance — and additionally suggests the metric is substantially measuring
pooled input statistics rather than learned reconstruction. The existing caveats on this metric
still apply (time-flattened CKA with D ≫ N; cosine includes additive learned prompts).

## Deviations and caveats

1. **Commit provenance.** Arms A and B ran at project commit `ebbcb4e`; arms C and D at
   `73a2769`, which adds `lenmod/ablate.py` and the `--variant frozen_dense|pool_fixed`
   branches to `run.py`. The edits are additive and the baseline/invariant code paths are
   untouched, but that is argued from the diff, not demonstrated by a re-run. The check
   that would close it: re-run baseline seed 1 at ℓ=200 under `73a2769` and confirm it is
   byte-identical to the existing record (six-case 64.74). ~25 min, not yet done.
2. **Arm D's checkpoints select late.** Best epochs [np.int64(197), np.int64(132), np.int64(158)] against
   [np.int64(84), np.int64(29), np.int64(139)] for A and [np.int64(85), np.int64(103), np.int64(49)] for C. Seed 1 selects epoch 197 of 200,
   i.e. at the cap, so arm D is the one arm where the 200-epoch budget may still be
   truncating. Any further budget would plausibly help D, not hurt it, so this does not
   threaten the direction of the result.
3. **Arm C's freeze is verified, not assumed.** `run.py` fingerprints the nine time-mixing
   layers (sha256 over parameter bytes) before and after backbone transfer and asserts
   equality, then freezes them; `pipeline_checks.frozen_time_mixer_parameters` records
   262,950 in all three seeds. A clean exit is the verification.
4. **Scope.** ℓ=200 only. These arms were not run at ℓ=50, ℓ=100 or native, so "unnecessary"
   is established at this rung and not across the ladder.
5. `missing_type_prompt_all_zeros: true` in all six new runs, as in every other run (F3).
