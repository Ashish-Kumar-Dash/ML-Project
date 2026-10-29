#!/usr/bin/env python3
"""Generation-fidelity audit on converged checkpoints.  Author: Ash (evaluation & analysis).

Re-scores the representations that lenmod/hooks.py (Sujal) already compares, with the
metrics the SOP specified, and first proves it is scoring the same thing:

  0. VERIFY      recompute the stored metric (lenmod.hooks.linear_cka on time-flattened
                 (N, 30*T) features + position-wise cosine) and assert it reproduces
                 results.json within TOL for every case and modality.  Mismatch -> stop.
  1. pooled CKA  linear CKA after masked time-mean pooling (frames where the TRUE modality is
                 non-padded), as the SOP specifies.  d = 30 << N = 686, so no D>N bias.
  2. dCKA flat   debiased linear CKA (unbiased HSIC, Song et al. 2012) on the same flattened
                 features as the stored metric, to measure how much D>N bias the stored
                 number carries (D = 1,500 to 13,125 vs N = 686).
  2b. padding test  flattened features encode WHERE the padding is (MOSI is left-padded,
                 ~86-90% of a/v frames), and padding layout = clip duration, shared across
                 modalities (r~0.99).  cka_flat_validonly zeroes truth-padded positions in both
                 gen and ref before the stored flattened CKA; cka_gen_vs_mask / cka_ref_vs_mask
                 measure how much of each flattened representation the 0/1 padding mask explains.
  3. label-direction fidelity (SOP "projection onto the component most predictive of the
                 label"): ridge readout fitted on TRAIN-split real representations (pooled,
                 standardised, alpha=1) -> sentiment.  On test, s_ref = w.ref, s_gen = w.gen.
                 Report r(s_gen, s_ref), r(s_ref, y), r(s_gen, y).

Representations are built exactly as in lenmod/hooks.py: ref = proj_m(x_m) + prompt{m}_nm,
gen = hooked time-mixer output^T + prompt{m}_m.  Model rebuilt exactly as run.py builds it
(PromptModel -> swap_time_mixers for invariant), checkpoint loaded with strict=True.
No fallbacks: any missing key, shape mismatch, epoch mismatch or verification failure raises.
Output: results/fidelity_audit.json
"""
import argparse, hashlib, json, re, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import lenmod.compat  # noqa: F401,E402  upstream path + torch.load shim
from lenmod.config import make_hp  # noqa: E402
from lenmod.hooks import CASE_MODALITY_MAP, GenerationHookManager, _unpack_batch, cosine_similarity_fidelity, linear_cka  # noqa: E402
from lenmod.operator import swap_time_mixers  # noqa: E402
try:  # arms C/D exist from project commit 73a2769; absence is only an error if such a run is scored
    from lenmod.ablate import swap_fixed_pool  # noqa: E402
except ImportError:
    swap_fixed_pool = None
from src.model import PromptModel  # noqa: E402
from src.mosidata import MOSIData  # noqa: E402

TOL = 5e-3
NUM_HEADS = 2  # run.py default; worker scripts never pass --num_heads. Verified indirectly by step 0.


def sha8(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:8]


def unbiased_hsic(K, L):
    n = K.shape[0]
    K = K.copy(); L = L.copy(); np.fill_diagonal(K, 0); np.fill_diagonal(L, 0)
    one = np.ones(n)
    t1 = np.sum(K * L)
    t2 = (one @ K @ one) * (one @ L @ one) / ((n - 1) * (n - 2))
    t3 = 2.0 / (n - 2) * (one @ K @ L @ one)
    return (t1 + t2 - t3) / (n * (n - 3))


def debiased_cka(X, Y):
    X = X.reshape(len(X), -1).astype(np.float64); Y = Y.reshape(len(Y), -1).astype(np.float64)
    K, L = X @ X.T, Y @ Y.T
    return float(unbiased_hsic(K, L) / np.sqrt(unbiased_hsic(K, K) * unbiased_hsic(L, L)))


def linear_cka_np(X, Y):
    X = X - X.mean(0); Y = Y - Y.mean(0)
    return float(np.linalg.norm(Y.T @ X) ** 2 / (np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y)))


def pearson(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def masked_pool(rep, mask):
    """rep (N, d, T), mask (N, T) bool -> (N, d) mean over valid frames."""
    m = mask[:, None, :].astype(np.float64)
    return (rep * m).sum(-1) / m.sum(-1)


def build(run_dir, d, device):
    sl, fd, hp_ = d["sequence_lengths"], d["feature_dimensions"], d["hyperparameters"]
    hp = make_hp(L=sl["L"], A=sl["A"], V=sl["V"], orig_dims=[fd["d_l"], fd["d_a"], fd["d_v"]], dataset=d["dataset"],
                 drop_rate=hp_["drop_rate"], batch_size=hp_["batch_size"], lr=hp_["lr"], num_epochs=hp_["num_epochs"],
                 seed=hp_["seed"], name=str(run_dir / "checkpoint.pt"), data_path=d["data_path"],
                 pretrained_model=d["pretrained_model"], use_cuda=device.type == "cuda")
    model = PromptModel(hp)
    # mirrors run.py step 2 exactly; frozen_dense (arm C) is the baseline architecture with frozen weights
    expected_op = {"baseline": "mlp_dense", "invariant": "cross_attn",
                   "frozen_dense": "mlp_dense_frozen_at_init", "pool_fixed": "avg_pool_fixed_zero_param"}
    assert d["operator"] == expected_op[d["variant"]], (d["variant"], d["operator"])
    if d["variant"] == "invariant":
        model = swap_time_mixers(model, operator=d["operator"], num_heads=NUM_HEADS)
    elif d["variant"] == "pool_fixed":
        assert swap_fixed_pool is not None, "pool_fixed run but lenmod.ablate unavailable"
        model = swap_fixed_pool(model)
    ckpt = torch.load(run_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert ckpt["epoch"] == d["training_outcome"]["best_epoch"], (ckpt["epoch"], d["training_outcome"]["best_epoch"])
    model.load_state_dict(ckpt["state_dict"], strict=True)
    assert (model.missing_type_prompt == 0).all()
    return model.to(device).eval()


@torch.no_grad()
def refs(model, loader, device):
    out = {"l": [], "a": [], "v": []}; mask = {"l": [], "a": [], "v": []}; ys = []
    for batch in loader:
        t, a, v = _unpack_batch(batch, device)
        out["l"].append((model.proj_l(t.transpose(1, 2)) + model.promptl_nm).cpu())
        out["a"].append((model.proj_a(a.transpose(1, 2)) + model.prompta_nm).cpu())
        out["v"].append((model.proj_v(v.transpose(1, 2)) + model.promptv_nm).cpu())
        for k, x in (("l", t), ("a", a), ("v", v)):
            mask[k].append((x.abs().sum(-1) != 0).cpu())
        ys.append(batch[1].reshape(-1).cpu())
    return ({k: torch.cat(v).double().numpy() for k, v in out.items()},
            {k: torch.cat(v).numpy() for k, v in mask.items()}, torch.cat(ys).double().numpy())


@torch.no_grad()
def audit_run(run_dir, device):
    d = json.loads((run_dir / "results.json").read_text())
    stored = d["test_results"]["generation_fidelity"]["cases"]
    model = build(run_dir, d, device)
    hp_ = d["hyperparameters"]
    mk = lambda split: DataLoader(MOSIData(d["data_path"], split_type=split, drop_rate=hp_["drop_rate"]),
                                  batch_size=hp_["batch_size"], shuffle=False)
    test_loader = mk("test")
    ref_tr, mask_tr, y_tr = refs(model, mk("train"), device)
    ref_te, mask_te, y_te = refs(model, test_loader, device)
    assert len(y_tr) == 1284 and len(y_te) == 686
    # ridge readouts on TRAIN real representations
    readout = {}
    for m in "lav":
        keep = mask_tr[m].any(1)
        X = masked_pool(ref_tr[m][keep], mask_tr[m][keep]); mu, sd = X.mean(0), X.std(0) + 1e-8
        Z = (X - mu) / sd
        w = np.linalg.solve(Z.T @ Z + 1.0 * np.eye(Z.shape[1]), Z.T @ (y_tr[keep] - y_tr[keep].mean()))
        readout[m] = (mu, sd, w)
    cases, max_dev = {}, 0.0
    for code, spec in CASE_MODALITY_MAP.items():
        cres = {}
        with GenerationHookManager(model) as hm:
            for batch in test_loader:
                t, a, v = _unpack_batch(batch, device)
                model(t, a, v, [code] * t.size(0))
            for m, layer in spec["layers"].items():
                g = hm.get_layer_output(layer)
                assert g is not None, f"{run_dir.name}: hook on {layer} captured nothing"
                gen = g.transpose(1, 2) + getattr(model, f"prompt{m}_m").detach().cpu()
                ref = torch.from_numpy(ref_te[m]).float()
                assert gen.shape == ref.shape, (layer, tuple(gen.shape), tuple(ref.shape))
                # 0. verification against stored values
                cos = cosine_similarity_fidelity(gen, ref, dim=1)["mean_cosine_sim"]
                cka_flat = linear_cka(gen, ref)
                s = stored[str(code)]["modalities"][m]
                dev = max(abs(cos - s["cosine_similarity"]), abs(cka_flat - s["linear_cka"]))
                assert dev < TOL, f"{run_dir.name} case {code} {m}: recomputed differs from results.json by {dev:.2e}"
                max_dev = max(max_dev, dev)
                gen_np, ref_np = gen.double().numpy(), ref_te[m]
                mk_t = torch.from_numpy(mask_te[m]).float()[:, None, :]  # (N, 1, T) truth validity
                keep = mask_te[m].any(1)
                gp, rp = masked_pool(gen_np[keep], mask_te[m][keep]), masked_pool(ref_np[keep], mask_te[m][keep])
                mu, sd, w = readout[m]
                s_gen, s_ref, yk = ((gp - mu) / sd) @ w, ((rp - mu) / sd) @ w, y_te[keep]
                cres[m] = {"layer": layer, "D_flat": int(gen_np[0].size), "N": int(keep.sum()),
                           "stored_cka_flat": s["linear_cka"], "stored_cosine": s["cosine_similarity"],
                           "recomputed_cka_flat": cka_flat, "recomputed_cosine": cos,
                           "dcka_flat": debiased_cka(gen_np, ref_np),
                           "cka_pooled": linear_cka_np(gp, rp),
                           "cka_flat_validonly": linear_cka(gen * mk_t, ref * mk_t),
                           "cka_gen_vs_mask": linear_cka(gen, mk_t[:, 0, :]),
                           "cka_ref_vs_mask": linear_cka(ref, mk_t[:, 0, :]),
                           "label_dir_r_gen_ref": pearson(s_gen, s_ref),
                           "label_dir_r_ref_y": pearson(s_ref, yk), "label_dir_r_gen_y": pearson(s_gen, yk)}
        cases[str(code)] = {"missing": spec["missing"], "modalities": cres}
    return {"run": run_dir.name, "variant": d["variant"], "operator": d["operator"], "seed": hp_["seed"],
            "sequence_lengths": d["sequence_lengths"], "best_epoch": d["training_outcome"]["best_epoch"],
            "checkpoint_sha256_prefix": sha8(run_dir / "checkpoint.pt"), "data_path": d["data_path"],
            "six_case_acc2": d["test_results"]["mean_cases_0_to_5"]["acc2"],
            "per_case_acc2": {k: v["acc2"] for k, v in d["test_results"]["per_case"].items()},
            "verification_max_abs_dev": max_dev, "cases": cases}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", default="runs_e200")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/fidelity_audit.json")
    ap.add_argument("--per_run_dir", default="results/fidelity_audit_runs",
                    help="resumable per-run records; use a fresh dir for a new run root so records never collide")
    ap.add_argument("--pattern", default=r"^mosi_ladder_(l50|l100|l200|native)_(baseline|invariant)_seed\d+$",
                    help="run-dir name regex; default = the A/B ladder")
    args = ap.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        assert torch.cuda.is_available(), "CUDA requested but unavailable (no silent CPU fallback)"
    # Scope set by --pattern (default: A/B ladder).  Non-matching run dirs are listed as skipped, never scored.
    pat = re.compile(args.pattern)
    cand = sorted(p for p in (REPO / args.runs_root).iterdir() if p.is_dir() and p.name != "logs")
    runs = [p for p in cand if pat.match(p.name) and (p / "results.json").exists() and (p / "checkpoint.pt").exists()]
    skipped = [p.name for p in cand if p not in runs]
    assert runs, "no runs with results.json + checkpoint.pt"
    out = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "script": "tools/fidelity_audit.py",
           "torch": torch.__version__, "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
           "tolerance_vs_results_json": TOL, "num_heads_assumed": NUM_HEADS, "pattern": args.pattern, "skipped_dirs": skipped, "runs": []}
    per_run_dir = REPO / args.per_run_dir; per_run_dir.mkdir(parents=True, exist_ok=True)
    for rd in runs:
        pj = per_run_dir / f"{rd.name}.json"
        if pj.exists():  # resume: per-run record already verified and written
            r = json.loads(pj.read_text()); out["runs"].append(r); print(f"[resume] {rd.name}", flush=True); continue
        torch.manual_seed(0)
        r = audit_run(rd, device)
        pj.write_text(json.dumps(r, indent=2))
        assert r["sequence_lengths"]["A"] in (50, 100, 200, 375) and json.loads((rd / "results.json").read_text())["hyperparameters"]["num_epochs"] == 200
        out["runs"].append(r)
        print(f"[ok] {rd.name}: verified (max dev {r['verification_max_abs_dev']:.1e})", flush=True)
        torch.cuda.empty_cache() if device.type == "cuda" else None
    (REPO / args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out} with {len(out['runs'])} runs")


if __name__ == "__main__":
    main()
