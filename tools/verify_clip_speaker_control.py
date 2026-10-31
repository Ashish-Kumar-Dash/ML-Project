
"""Speaker-identity control for the CLIP checks: per-video centring (label-free) of clip-level features.
MOSI splits are speaker-independent (one speaker per video), so per-video centring removes speaker identity."""
import sys, json, pickle, numpy as np
from collections import defaultdict
from sklearn.linear_model import Ridge
from sklearn.cross_decomposition import PLSRegression
from scipy.stats import pearsonr
rng = np.random.default_rng(0)
ref = pickle.load(open("data/mosi_data.pkl", "rb"), encoding="latin1"); nat = pickle.load(open("data/mosi_clip_b32_native.pkl", "rb"))
S = ("train", "valid", "test")
y = {s: np.asarray(ref[s]["labels"]).reshape(len(ref[s]["labels"]), -1)[:, 0].astype(float) for s in S}
vid = {s: np.array([r[0].decode().rsplit("_", 1)[0] for r in ref[s]["id"]]) for s in S}
def mm(X):
    X = np.asarray(X, dtype=np.float64); X = np.where(np.isneginf(X), 0, X); m = (np.abs(X).sum(-1) > 0)[..., None]
    return (X * m).sum(1) / np.maximum(m.sum(1), 1)
raw = {s: {"CLIP 512-d": np.stack([f.astype(np.float32).mean(0) for f in nat["frames"][s]]).astype(np.float64),
           "Facet 20-d": mm(ref[s]["vision"]), "COVAREP 5-d": mm(ref[s]["audio"]), "GloVe 300-d": mm(ref[s]["text"])} for s in S}
def centre(X, v):
    X = X.copy()
    for u in np.unique(v): X[v == u] -= X[v == u].mean(0)
    return X
out = {"videos_per_split": {s: int(len(np.unique(vid[s]))) for s in S},
       "speaker_overlap_train_test": int(len(set(vid["train"]) & set(vid["test"])))}
# how much of each feature's variance is between-video (speaker) variance
out["between_video_variance_share"] = {k: float(1 - centre(raw["test"][k], vid["test"]).var(0).sum() / raw["test"][k].var(0).sum()) for k in raw["test"]}
def zs(tr, X): mu, sd = tr.mean(0), tr.std(0); sd[sd == 0] = 1; return (X - mu) / sd
ywithin = y["test"] - np.array([y["test"][vid["test"] == u].mean() for u in vid["test"]])
alphas = [10.0 ** k for k in range(-2, 7)]
for tag, feats in (("raw", raw), ("video_centred", {s: {k: centre(v, vid[s]) for k, v in raw[s].items()} for s in S})):
    pr = {}
    for name in feats["train"]:
        Xtr, Xva, Xte = (zs(feats["train"][name], feats[s][name]) for s in S)
        a = max(alphas, key=lambda a: pearsonr(Ridge(alpha=a).fit(Xtr, y["train"]).predict(Xva), y["valid"])[0])
        p = Ridge(alpha=a).fit(Xtr, y["train"]).predict(Xte)
        b = [pearsonr(p[i], y["test"][i])[0] for i in (rng.integers(0, len(p), len(p)) for _ in range(1000))]
        pr[name] = {"alpha": a, "alpha_at_grid_edge": a in (alphas[0], alphas[-1]), "test_r": float(pearsonr(p, y["test"])[0]),
                    "test_r_ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
                    "test_r_within_video": float(pearsonr(p, ywithin)[0])}
    out[f"probe_{tag}"] = pr
def cka(X, Y):
    X = X - X.mean(0); Y = Y - Y.mean(0); return float(np.linalg.norm(X.T @ Y) ** 2 / (np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y)))
ck = {}
T = {k: centre(zs(raw["train"][k], raw["test"][k]), vid["test"]) for k in raw["test"]}
for a in ("Facet 20-d", "CLIP 512-d"):
    for b in ("GloVe 300-d", "COVAREP 5-d"):
        v = cka(T[a], T[b])
        # null: permute clips WITHIN video, which preserves the centring and the speaker structure
        null = []
        for _ in range(500):
            idx = np.arange(len(T[a]))
            for u in np.unique(vid["test"]):
                w = np.flatnonzero(vid["test"] == u); idx[w] = rng.permutation(w)
            null.append(cka(T[a][idx], T[b]))
        ck[f"{a} | {b}"] = {"cka": v, "null_mean": float(np.mean(null)), "null95": float(np.percentile(null, 95)),
                            "excess_over_null": v - float(np.mean(null)), "p_perm": float((np.sum(np.array(null) >= v) + 1) / 501)}
out["intermodal_cka_test_video_centred"] = ck
json.dump(out, open("results/clip_feature_checks_speaker_control.json", "w"), indent=1)
print(json.dumps(out, indent=1))
