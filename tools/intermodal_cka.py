#!/usr/bin/env python3
"""Inter-modal linear CKA along the pooled length ladder (SOP Sec. IV, "report measured
CKA at each rung to demonstrate rather than assert that correlation is approximately
constant").  Author: Ash (evaluation & analysis).

Input-feature measurement only: no model, no training.  Assert-on-mismatch by design:
  * every rung file must match data/ladder/SHA256SUMS, else the script stops;
  * shapes are asserted against the expected (N, T, d) per rung.

Two readings per rung:
  clip-level   time-mean of each modality over its own non-padded frames (zero rows are
               padding; ~86-90% of audio/vision frames).  This is the SOP's mean-pooled
               measure.  Average pooling preserves the per-clip mean up to boundary bins,
               so this reading is expected to be ~constant by construction.
  frame-level  audio-vision CKA over individual time steps where both are non-padded.
               Defined only where A == V (l50/l100/l200); n/a at native (375 vs 500).
               This is the reading that pooling can change (temporal smoothing).
Also reported: the unmasked time-mean (includes padding zeros).  It is confounded by clip
duration, which is shared across modalities, and is given for transparency only.

Statistics (revised after a first pass): one COVAREP dimension carries 91% of pooled audio
variance, so raw linear CKA mostly tracks that single dimension; every reading is therefore
reported raw AND per-dimension z-scored, plus debiased CKA (unbiased HSIC, Song et al. 2012)
on the z-scored features.  Chance level is a permutation null (pairing shuffled across clips,
200 permutations, seed 0; 95th percentile).  A percentile bootstrap was dropped: resampling
duplicates inflate CKA, so its interval excluded its own point estimate.
Output: results/intermodal_cka.json (+ .txt).
"""
import hashlib, json, pickle, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
LADDER = REPO / "data" / "ladder"
RUNGS = {  # name: (file, expected A, expected V)
    "l50": ("mosi_data_noalign_l50.pkl", 50, 50),
    "l100": ("mosi_data_noalign_l100.pkl", 100, 100),
    "l200": ("mosi_data_noalign_l200.pkl", 200, 200),
    "native": ("mosi_data_noalign_native.pkl", 375, 500),
}
SPLITS = ("train", "valid", "test")
N_SPLIT = {"train": 1284, "valid": 229, "test": 686}
DIMS = {"text": 300, "audio": 5, "vision": 20}
NPERM, SEED = 200, 0


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def cka_feat(X, Y):
    """Linear CKA via the feature-space identity (valid for any N, cheap when d is small)."""
    X = X - X.mean(0, keepdims=True); Y = Y - Y.mean(0, keepdims=True)
    num = np.linalg.norm(Y.T @ X, "fro") ** 2
    den = np.linalg.norm(X.T @ X, "fro") * np.linalg.norm(Y.T @ Y, "fro")
    assert den > 0, "degenerate representation"
    return float(num / den)


def masked_mean(x):
    valid = np.abs(x).sum(-1) != 0  # (N, T)
    cnt = valid.sum(1)
    return (x * valid[..., None]).sum(1) / np.maximum(cnt, 1)[:, None], cnt, valid


def z(X):
    sd = X.std(0); keep = sd > 0
    return (X[:, keep] - X[:, keep].mean(0)) / sd[keep]


def unbiased_hsic(K, L):
    n = K.shape[0]; K = K.copy(); L = L.copy(); np.fill_diagonal(K, 0); np.fill_diagonal(L, 0)
    one = np.ones(n)
    return (np.sum(K * L) + (one @ K @ one) * (one @ L @ one) / ((n - 1) * (n - 2)) - 2.0 / (n - 2) * (one @ K @ (L @ one))) / (n * (n - 3))


def dcka(X, Y):
    K, L = X @ X.T, Y @ Y.T
    return float(unbiased_hsic(K, L) / np.sqrt(unbiased_hsic(K, K) * unbiased_hsic(L, L)))


def perm_null(X, Y, rng):
    v = [cka_feat(X, Y[rng.permutation(len(Y))]) for _ in range(NPERM)]
    return float(np.percentile(v, 95))


def readings(X, Y, rng, debiased=True):
    Xz, Yz = z(X), z(Y)
    r = {"cka_raw": cka_feat(X, Y), "cka_z": cka_feat(Xz, Yz), "perm95_cka_z": perm_null(Xz, Yz, rng)}
    if debiased:
        r["dcka_z"] = dcka(Xz, Yz)
    return r


def main():
    sums = dict(reversed(l.split()) for l in (LADDER / "SHA256SUMS").read_text().split("\n") if l.strip())
    out = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "script": "tools/intermodal_cka.py",
           "null": {"permutations": NPERM, "seed": SEED, "stat": "95th percentile of cka_z under shuffled pairing"}, "rungs": {}}
    for name, (fn, A, V) in RUNGS.items():
        p = LADDER / fn
        digest = sha256(p)
        assert digest == sums[fn], f"{fn}: sha256 {digest[:8]} != SHA256SUMS {sums[fn][:8]}"
        d = pickle.load(open(p, "rb"))
        feats = {}
        for m, T in (("text", 50), ("audio", A), ("vision", V)):
            parts = []
            for s in SPLITS:
                x = np.asarray(d[s][m], dtype=np.float64)
                assert x.shape == (N_SPLIT[s], T, DIMS[m]), f"{name}/{s}/{m} shape {x.shape}"
                assert np.isfinite(x).all(), f"{name}/{s}/{m} non-finite"
                parts.append(x)
            feats[m] = np.concatenate(parts, 0)
        N = len(feats["text"])
        rng = np.random.default_rng(SEED)
        pooled, counts, valid = {}, {}, {}
        for m, x in feats.items():
            pooled[m], counts[m], valid[m] = masked_mean(x)
        keep = np.all([counts[m] > 0 for m in feats], axis=0)  # clips with >=1 real frame in every modality
        r = {"file": fn, "sha256_prefix": digest[:8], "N_clips": int(N), "N_clips_all_modalities_present": int(keep.sum()),
             "padding_fraction": {m: float(1 - valid[m].mean()) for m in feats}, "clip_level_masked": {}, "clip_level_unmasked": {}}
        for a, b in (("text", "audio"), ("text", "vision"), ("audio", "vision")):
            r["clip_level_masked"][f"{a}-{b}"] = readings(pooled[a][keep], pooled[b][keep], rng)
            r["clip_level_unmasked"][f"{a}-{b}"] = readings(feats[a].mean(1), feats[b].mean(1), rng, debiased=False)
        r["valid_frame_count_corr_audio_vision"] = float(np.corrcoef(counts["audio"], counts["vision"])[0, 1])
        if A == V:
            both = valid["audio"] & valid["vision"]
            Xa, Xv = feats["audio"][both], feats["vision"][both]
            fl = readings(Xa, Xv, rng, debiased=False); fl["N_frames"] = int(both.sum())
            r["frame_level_audio_vision"] = fl
        else:
            r["frame_level_audio_vision"] = None
        out["rungs"][name] = r
        print(f"[{name}] sha {digest[:8]} ok", flush=True)
        del d, feats
    res = REPO / "results"
    (res / "intermodal_cka.json").write_text(json.dumps(out, indent=2))
    lines = ["rung    pair            masked: raw    z     dCKA_z  null95 | unmasked: raw   z"]
    for n_, r in out["rungs"].items():
        for pair, v in r["clip_level_masked"].items():
            u = r["clip_level_unmasked"][pair]
            lines.append(f"{n_:7s} {pair:15s} {v['cka_raw']:.4f} {v['cka_z']:.4f} {v['dcka_z']:.4f} {v['perm95_cka_z']:.4f} | {u['cka_raw']:.4f} {u['cka_z']:.4f}")
        fl = r["frame_level_audio_vision"]
        lines.append(f"{n_:7s} frame a-v       " + (f"{fl['cka_raw']:.4f} {fl['cka_z']:.4f}   --   {fl['perm95_cka_z']:.4f}  N_frames={fl['N_frames']}" if fl else "n/a (A != V)"))
        lines.append(f"{n_:7s} corr(valid-frame count a, v) = {r['valid_frame_count_corr_audio_vision']:.3f}")
    (res / "intermodal_cka.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
