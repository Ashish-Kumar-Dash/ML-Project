# Length-Invariant Missing-Modality Generation

Diagnosing and repairing the generation module of **MPLMM** ([Guo, Jin & Zhao, ACL 2024](https://aclanthology.org/2024.acl-long.94)).

---

## 1. The Core Finding: Quadratic Parameter Explosion

MPLMM restores multimodal sentiment analysis under absent modalities by freezing a pretrained [MulT](https://github.com/yaohungt/Multimodal-Transformer) backbone and training three lightweight prompt types alongside a missing-modality generation module. The paper's Limitations section attributes anticipated degradation on raw features to weaker inter-modal correlation and leaves it as future work.

Analysis of the official implementation reveals an unreported architectural property: the generation module's time-mixing layer (`MLPLayer`) applies a `Conv1d(kernel_size=1)` **after a transpose**, placing the temporal sequence dimension into the convolution's channel slot. The kernel operates across time without translation equivariance or weight sharing. Across the 9 time-mixing layers, parameter counts scale bilinearly with input and output sequence lengths:

$$\text{params} = \ell_{\text{out}} \times (\ell_p + \sum \ell_{\text{in}}) + \ell_{\text{out}}$$

| CMU-MOSI Configuration | Sequence Lengths ($L/A/V$) | 9 Time-Mixing Layers | All Length-Dependent Params | Trainable Params (vs 1.07M Backbone) |
| :--- | :---: | :---: | :---: | :---: |
| **Aligned** | $50 / 50 / 50$ | 37,650 | 55,650 | 88,141 *(8.2%)* |
| **Unaligned (Native)** | $50 / 375 / 500$ | **972,175** | **1,083,175** | **1,115,666 — 104% (Explosion!)** |

At unaligned sequence lengths, the length-dependent generation parameters exceed the entire frozen backbone they were designed to avoid fine-tuning.

---

## 2. The Solution: Length-Invariant Generator (`lenmod.operator`)

We implement [`LengthInvariant`](lenmod/operator.py), a cross-attention generator operator that completely decouples parameter counts from sequence lengths:
- Uses a fixed, canonical learned query vector ($K = 50$) and interpolates along the temporal axis to target $\ell_{\text{out}}$ when needed.
- **Strictly Constant Parameter Count**: Exactly **46,980 parameters across all 9 layers** (5,220 parameters per layer) regardless of sequence lengths.
- Achieves a **95.2% parameter reduction** on unaligned MOSI ($46,980$ vs $972,175$).
- Preserves channel-mixing projections as `MLPLayer` while dynamically replacing time-mixers via [`swap_time_mixers`](lenmod/operator.py).

---

## 3. Empirical 12 GB GPU OOM Benchmark

Upstream defaulted to `--batch_size 64` without gradient accumulation. Using [`tools/measure_peak_memory.py`](tools/measure_peak_memory.py) on an NVIDIA GeForce RTX 3060 (11.63 GiB VRAM), we empirically tested peak CUDA memory allocation:

| Model Variant | Batch Size | Static Model Memory | Forward Peak Memory | Backward Peak Memory | Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Baseline (Dense)** | 16 | 0.008 GiB | 3.712 GiB | **3.809 GiB** | **SUCCESS** *(~7.8 GiB headroom)* |
| **Baseline (Dense)** | 64 | 0.008 GiB | **N/A** | **N/A** | **OOM in Forward Pass** |
| **Invariant (Cross-Attn)** | 16 | 0.005 GiB | 3.736 GiB | **3.832 GiB** | **SUCCESS** *(~7.8 GiB headroom)* |
| **Invariant (Cross-Attn)** | 64 | 0.005 GiB | **N/A** | **N/A** | **OOM in Forward Pass** |

**Remedy**: Training with gradient accumulation (`batch_size=16`, `accum_steps=4`, effective batch size 64) is strictly required to execute the paper's optimization dynamics within a 12 GB GPU envelope.

---

## 4. Sequence Ladder Pricing: 1-Epoch Training Duration Benchmark

Using [`tools/benchmark_epoch_timing.py`](tools/benchmark_epoch_timing.py), we timed 1 full training epoch (1,284 samples, 81 batches, `batch_size=16`, `accum_steps=4`) on an NVIDIA GeForce RTX 3060 across all 4 sequence ladder lengths. Timings reflect the mean ± standard deviation over 3 steady-state epochs after 1 warmup epoch:

| Ladder Rung | Sequence Lengths ($L/A/V$) | Variant | Time-Mixing Params | Train Epoch (s) | Throughput | Peak VRAM | 40-Epoch Runtime |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **L50** | $50 / 50 / 50$ | Dense | 37,650 | 4.108s ± 0.28s | 312.5 smp/s | 0.19 GiB | 2.9 min |
| **L50** | $50 / 50 / 50$ | Invariant | **46,980** | 4.871s ± 0.29s | 263.6 smp/s | 0.19 GiB | 3.5 min |
| **L100** | $50 / 100 / 100$ | Dense | 92,750 | 4.198s ± 0.38s | 305.8 smp/s | 0.39 GiB | 3.0 min |
| **L100** | $50 / 100 / 100$ | Invariant | **46,980** | 4.987s ± 0.11s | 257.5 smp/s | 0.39 GiB | 3.6 min |
| **L200** | $50 / 200 / 200$ | Dense | 262,950 | 6.291s ± 0.05s | 204.1 smp/s | 0.98 GiB | 4.6 min |
| **L200** | $50 / 200 / 200$ | Invariant | **46,980** | 6.766s ± 0.07s | 189.8 smp/s | 0.99 GiB | 4.9 min |
| **Native** | $50 / 375 / 500$ | Dense | 972,175 *(Explosion)* | 16.699s ± 0.27s | 76.9 smp/s | 3.56 GiB | 12.0 min |
| **Native** | $50 / 375 / 500$ | Invariant | **46,980** *(-95.2%)* | 17.404s ± 0.13s | 73.8 smp/s | 3.57 GiB | 12.5 min |

### Pricing Relative Overhead:
| Ladder Rung | Sequence Lengths | Dense Epoch | Invariant Epoch | Relative Overhead | Parameter Savings |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **L50** | $50 / 50 / 50$ | 4.108s | 4.871s | 1.19x (+0.76s) | Invariant +24.8% params ($K=50$ canonical query) |
| **L100** | $50 / 100 / 100$ | 4.198s | 4.987s | 1.19x (+0.79s) | Invariant **-49.3%** params |
| **L200** | $50 / 200 / 200$ | 6.291s | 6.766s | 1.08x (+0.48s) | Invariant **-82.1%** params |
| **Native** | $50 / 375 / 500$ | 16.699s | 17.404s | **1.04x (+0.70s, only 4.2%!)** | Invariant **-95.2%** params ($46,980$ vs $972,175$) |

> [!TIP]
> **Ladder Pricing Finding**: While cross-attention incurs ~19% overhead on short sequences due to attention matrices, this overhead asymptotically shrinks to just **4.2%** on native unaligned sequences ($50/375/500$). The invariant generator completely avoids the 95.2% parameter explosion without any meaningful compute penalty.

---

## 5. Experimental Results

All experiments were trained for 40 epochs on an RTX 3060, reloaded their lowest validation loss checkpoint, and evaluated strictly on the test set across all 7 modality conditions ($0 \dots 6$). Every run records upstream commit hash `c6f8d18e9222bd65165a3214820201dc51a35f19` in structured JSON:

| Run Identifier | Dataset | Sequence Lengths ($L/A/V$) | 9-Layer Parameters | Best Val Loss | Mean Acc-2 (Cases 0–5) | Complete Data (Case 6) | Cosine Fidelity | Linear CKA | Result Artifact |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Aligned Baseline** | `mosi_data.pkl` | $50/50/50$ | 37,650 | 1.1973 | 59.78% | 60.21% | 0.0714 | 0.3988 | [`runs/mosi_baseline_seed1/results.json`](runs/mosi_baseline_seed1/results.json) |
| **Aligned Invariant** | `mosi_data.pkl` | $50/50/50$ | 46,980 | **1.0907** | **62.20%** *(+2.4%)* | **68.90%** *(+8.7%)* | **0.0944** | 0.0725 | [`runs/mosi_invariant_seed1/results.json`](runs/mosi_invariant_seed1/results.json) |
| **Ladder L100 Invariant** | `mosi_data_noalign_l100.pkl` | $50/100/100$ | **46,980** | 1.1153 | **67.30%** | 78.20% | 0.1653 | 0.1395 | [`runs/mosi_ladder_l100_invariant/results.json`](runs/mosi_ladder_l100_invariant/results.json) |
| **Ladder L200 Invariant** | `mosi_data_noalign_l200.pkl` | $50/200/200$ | **46,980** | **1.0924** | **67.15%** | 79.12% | 0.1055 | 0.1633 | [`runs/mosi_ladder_l200_invariant/results.json`](runs/mosi_ladder_l200_invariant/results.json) |
| **Unaligned Baseline** | `mosi_data_noalign.pkl` | $50/375/500$ | 972,175 | 1.0941 | 67.10% | 80.34% | 0.0208 | 0.2656 | [`runs/mosi_noalign_baseline_seed1/results.json`](runs/mosi_noalign_baseline_seed1/results.json) |
| **Unaligned Invariant** | `mosi_data_noalign.pkl` | $50/375/500$ | **46,980** *(-95.2%)* | 1.1283 | 66.31% | 78.66% | **0.1831** *(~9x)* | 0.1416 | [`runs/mosi_noalign_invariant_seed1/results.json`](runs/mosi_noalign_invariant_seed1/results.json) |

### Key Takeaways:
1. **Parameter Efficiency**: Invariant operator achieves **95.2% parameter reduction** on unaligned MOSI while matching performance within 0.79% and improving overall test loss ($1.1405$ vs $1.1821$).
2. **Aligned Gains**: On aligned MOSI, the invariant operator outperforms the dense baseline across all missing modality conditions (+2.42% mean Acc-2, +8.69% complete data).
3. **Sequence Ladder Stability**: Across the pooling ladder ($50 \rightarrow 100 \rightarrow 200 \rightarrow 375/500$), parameter counts remain strictly locked at 46,980 while accuracy remains steady ($\approx 66.3 - 67.3\%$).
4. **Generation Fidelity**: On unaligned sequences, the invariant generator produces nearly **9x higher cosine fidelity** to the true reference features (0.1831 vs 0.0208).

---

## 6. Repository Layout

```text
├── docs/
│   └── PROVENANCE.md              # SHA-256 hashes for datasets, backbones, and upstream code
├── data/
│   ├── SHA256SUMS                 # Data verification checksums
│   └── ladder/                    # Pooled ladder rungs (l50, l100, l200, native)
├── lenmod/                        # Runtime adaptation package (NO edits to upstream files)
│   ├── compat.py                  # sys.path injection and PyTorch >= 2.6 torch.load wrapper
│   ├── config.py                  # Complete Namespace builder (make_hp)
│   ├── hooks.py                   # PyTorch forward hooks, Cosine Fidelity, and Linear CKA
│   ├── invariant.py               # Time-mixer replacements
│   ├── operator.py                # LengthInvariant operator & swap_time_mixers
│   ├── trainer.py                 # Single-GPU fit, evaluate, and eval_cases loops
│   └── transfer.py                # Safe backbone weight transfer and parameter freezing
├── pretrained/
│   └── mosei.pt                   # Pretrained MOSEI aligned backbone checkpoint
├── runs/                          # JSON results tracking upstream commit hash
├── tests/                         # Pytest / Unittest test suite (21 tests passing)
│   ├── test_dead_prompts.py       # Zero-gradient trap verification on upstream prompts
│   ├── test_forward.py            # Forward pass across missing codes 0 to 6
│   ├── test_hooks.py              # Generation hook capture and CKA property tests
│   ├── test_ladder_benchmark.py   # Sequence ladder benchmark artifact & scaling test
│   ├── test_memory.py             # CUDA peak memory benchmark and OOM regression test
│   ├── test_operator.py           # Parameter constancy across lengths & gradient flow
│   ├── test_trainer.py            # Gradient accumulation and checkpoint reload tests
│   ├── test_transfer.py           # Weight transfer and parameter freeze tests
│   └── test_upstream_clean.py     # Verifies third_party/MPLMM submodule is pristine
├── third_party/
│   └── MPLMM/                     # Pristine upstream submodule (read-only, commit c6f8d18)
├── tools/
│   ├── benchmark_epoch_timing.py  # Ladder 1-epoch training timing benchmark
│   ├── count_params.py            # Detailed parameter breakdown
│   ├── make_fake_mosi.py          # Synthetic dataset generator for rapid testing
│   └── measure_peak_memory.py     # CUDA peak memory measurement tool
├── run.py                         # Strict 7-step build order experiment runner
└── pytest.ini                     # Pytest configuration
```

---

## 7. How to Run

### Run Unit Tests
```bash
pytest -v
```

### Run Peak Memory Benchmark (Confirm OOM)
```bash
python tools/measure_peak_memory.py --all
```

### Run Sequence Ladder Timing Benchmark
```bash
python tools/benchmark_epoch_timing.py --num_timed_epochs 3
```

### Run Training Experiments
```bash
# Aligned MOSI (Baseline vs Invariant)
python run.py --variant baseline --data_path data/mosi_data.pkl --num_epochs 40 --seed 1 --eval_fidelity --name runs/mosi_baseline_seed1
python run.py --variant invariant --data_path data/mosi_data.pkl --num_epochs 40 --seed 1 --eval_fidelity --name runs/mosi_invariant_seed1

# Unaligned MOSI (Baseline vs Invariant)
python run.py --variant baseline --data_path data/mosi_data_noalign.pkl --num_epochs 40 --seed 1 --eval_fidelity --name runs/mosi_noalign_baseline_seed1
python run.py --variant invariant --data_path data/mosi_data_noalign.pkl --num_epochs 40 --seed 1 --eval_fidelity --name runs/mosi_noalign_invariant_seed1

# Sequence Length Ladder Rungs
python run.py --variant invariant --data_path data/ladder/mosi_data_noalign_l100.pkl --num_epochs 40 --seed 1 --eval_fidelity --name runs/mosi_ladder_l100_invariant
python run.py --variant invariant --data_path data/ladder/mosi_data_noalign_l200.pkl --num_epochs 40 --seed 1 --eval_fidelity --name runs/mosi_ladder_l200_invariant
```

---

## 8. Project Status

- [x] Parameter audit analytical derivation and verification (`tools/count_params.py`)
- [x] Pretrained MOSEI aligned backbone (`runs/mosei_pretrain/RUN_RECORD.md`, `pretrained/mosei.pt`)
- [x] Baseline reproduction gate on aligned MOSI (`runs/mosi_gate/GATE_RECORD.md`)
- [x] Dead-prompt mathematical finding verification (`tests/test_dead_prompts.py`)
- [x] Runtime-adapted single-GPU trainer with gradient accumulation (`lenmod/trainer.py`)
- [x] Length-invariant cross-attention operator implementation (`lenmod/operator.py`)
- [x] Unit test suite verifying parameter constancy across 4 lengths (`tests/test_operator.py`)
- [x] Instrumented forward hooks capturing generation fidelity & CKA (`lenmod/hooks.py`)
- [x] 12 GB GPU OOM empirical proof and batch 16 sizing (`tools/measure_peak_memory.py`)
- [x] Sequence ladder training duration pricing benchmark across 4 lengths (`tools/benchmark_epoch_timing.py`)
- [x] Experimental validation across aligned, unaligned, and ladder datasets (`runs/`)

---

## 9. Team

- **Tanmay Jaiswal** (Member 1) — Feature pipeline and data engineering
- **Sujal Som** (Member 2) — Model and training
- **Ashish Kumar Dash** (Member 3) — Evaluation and analysis

ML course project, IIT Bhilai. Target paper: [MPLMM](https://github.com/zrguo/MPLMM). Backbone: [MulT](https://github.com/yaohungt/Multimodal-Transformer).
