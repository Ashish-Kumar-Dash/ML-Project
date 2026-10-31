
"""Step 3 verification of real CLIP ViT-B/32 features on CMU-MOSI (eval-only).
1. zero-shot sanity on real frames; 2. ridge sentiment probe CLIP vs Facet; 3. inter-modal CKA with permutation null."""
import sys, os, json, pickle, numpy as np, torch, torch.nn.functional as F
sys.path[:0] = ["/home/user/.claude-science-scratch/deps", "/home/user/.claude-science-scratch/tok", "."]
from sklearn.linear_model import Ridge
from sklearn.cross_decomposition import PLSRegression
from scipy.stats import pearsonr
from simple_tokenizer import SimpleTokenizer
rng = np.random.default_rng(0)
ref = pickle.load(open("data/mosi_data.pkl", "rb"), encoding="latin1")
nat = pickle.load(open("data/mosi_clip_b32_native.pkl", "rb"))
out = {"n_frames": {}}
def ids(s): return [r[0].decode() for r in ref[s]["id"]]
for s in ("train", "valid", "test"):
    assert nat["ids"][s] == ids(s), s
    nf = [len(f) for f in nat["frames"][s]]; out["n_frames"][s] = {"min": int(min(nf)), "median": float(np.median(nf)), "max": int(max(nf))}
y = {s: np.asarray(ref[s]["labels"]).reshape(len(ref[s]["labels"]), -1)[:, 0].astype(float) for s in ("train", "valid", "test")}
def masked_mean(X):  # mean over real (non-zero) rows; all-zero clips -> zeros
    X = np.asarray(X, dtype=np.float64); X = np.where(np.isneginf(X), 0, X); m = (np.abs(X).sum(-1) > 0)[..., None]
    return (X * m).sum(1) / np.maximum(m.sum(1), 1)
feats = {s: {"CLIP 512-d": np.stack([f.astype(np.float32).mean(0) for f in nat["frames"][s]]),
             "Facet 20-d": masked_mean(ref[s]["vision"]),
             "COVAREP 5-d (audio, ref)": masked_mean(ref[s]["audio"]),
             "GloVe 300-d (text, ref)": masked_mean(ref[s]["text"])} for s in ("train", "valid", "test")}
pls = PLSRegression(n_components=20).fit(feats["train"]["CLIP 512-d"], y["train"])
for s in feats: feats[s]["CLIP->PLS 20-d"] = pls.transform(feats[s]["CLIP 512-d"])
# 1. zero-shot (official tokenizer + text encoder)
tok = SimpleTokenizer("/home/user/.claude-science-scratch/tok/bpe_simple_vocab_16e6.txt.gz")
def tokenize(ts):
    sot, eot = tok.encoder["<|startoftext|>"], tok.encoder["<|endoftext|>"]; r = torch.zeros(len(ts), 77, dtype=torch.long)
    for k, t in enumerate(ts): i = [sot] + tok.encode(t) + [eot]; r[k, :len(i)] = torch.tensor(i)
    return r
full = torch.jit.load(os.path.expanduser("~/.cache/clip/ViT-B-32.pt"), map_location="cuda:0").eval()
def tenc(ts):
    with torch.no_grad(): return F.normalize(full.encode_text(tokenize(ts).cuda()).float(), dim=-1).cpu().numpy()
scene = ["a photo of a person talking to the camera", "a photo of a landscape", "a photo of food", "a photo of a car",
         "a screenshot of a document", "a photo of an animal", "a cartoon"]
mid = np.stack([f[len(f) // 2].astype(np.float32) for s in ("train", "valid", "test") for f in nat["frames"][s]])
win = (mid @ tenc(scene).T).argmax(1)
out["zero_shot_scene"] = {scene[k]: int((win == k).sum()) for k in range(len(scene))}
out["zero_shot_person_frac"] = float((win == 0).mean())
te = feats["test"]["CLIP 512-d"]; te = te / np.linalg.norm(te, axis=1, keepdims=True)
pos, neg = tenc(["a photo of a happy person", "a photo of a smiling person"]).mean(0), tenc(["a photo of a sad person", "a photo of an upset person"]).mean(0)
score = te @ pos - te @ neg; nz = y["test"] != 0
out["zero_shot_sentiment_test"] = {"pearson_r": float(pearsonr(score, y["test"])[0]), "p": float(pearsonr(score, y["test"])[1]),
    "acc2_non0_median_threshold": float(((score > np.median(score)) == (y["test"] > 0))[nz].mean())}
# 2. ridge probe, alpha picked on valid, scored on test (Acc-2 excludes zero labels; MulT 'non0' convention)
def zs(tr, X): mu, sd = tr.mean(0), tr.std(0); sd[sd == 0] = 1; return (X - mu) / sd
probe = {}
for name in feats["train"]:
    Xtr, Xva, Xte = (zs(feats["train"][name], feats[s][name]) for s in ("train", "valid", "test"))
    best = max([10.0 ** k for k in range(-2, 5)], key=lambda a: pearsonr(Ridge(alpha=a).fit(Xtr, y["train"]).predict(Xva), y["valid"])[0])
    p = Ridge(alpha=best).fit(Xtr, y["train"]).predict(Xte)
    r_, pv = pearsonr(p, y["test"])
    boots = []
    for _ in range(2000):
        i = rng.integers(0, len(p), len(p)); boots.append(pearsonr(p[i], y["test"][i])[0])
    probe[name] = {"alpha": best, "test_r": float(r_), "test_r_ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                   "acc2_non0": float(((p > 0) == (y["test"] > 0))[nz].mean()), "pred": p}
# paired bootstrap: CLIP vs Facet
pc, pf, yt = probe["CLIP 512-d"]["pred"], probe["Facet 20-d"]["pred"], y["test"]; d = []
for _ in range(2000):
    i = rng.integers(0, len(yt), len(yt)); d.append(pearsonr(pc[i], yt[i])[0] - pearsonr(pf[i], yt[i])[0])
out["probe_clip_minus_facet_r"] = {"diff": float(probe["CLIP 512-d"]["test_r"] - probe["Facet 20-d"]["test_r"]),
                                   "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], "frac_boot_le0": float(np.mean(np.array(d) <= 0))}
for v in probe.values(): v.pop("pred")
out["probe"] = probe
# 3. inter-modal linear CKA (clip-level, z-scored, test split) with a 500-permutation null
def cka(X, Y):
    X = X - X.mean(0); Y = Y - Y.mean(0); hsic = np.linalg.norm(X.T @ Y) ** 2
    return float(hsic / (np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y)))
T = feats["test"]; Z = {k: zs(feats["train"][k], T[k]) for k in T}
pairs = [("Facet 20-d", "GloVe 300-d (text, ref)"), ("CLIP 512-d", "GloVe 300-d (text, ref)"), ("CLIP->PLS 20-d", "GloVe 300-d (text, ref)"),
         ("Facet 20-d", "COVAREP 5-d (audio, ref)"), ("CLIP 512-d", "COVAREP 5-d (audio, ref)"), ("CLIP->PLS 20-d", "COVAREP 5-d (audio, ref)")]
ck = {}
for a, b in pairs:
    v = cka(Z[a], Z[b]); null = [cka(Z[a][rng.permutation(len(Z[a]))], Z[b]) for _ in range(500)]
    ck[f"{a} | {b}"] = {"cka": v, "null_mean": float(np.mean(null)), "null95": float(np.percentile(null, 95)), "excess_over_null": v - float(np.mean(null)),
                        "p_perm": float((np.sum(np.array(null) >= v) + 1) / 501)}
out["intermodal_cka_test"] = ck
json.dump(out, open("results/clip_feature_checks.json", "w"), indent=1)
print(json.dumps(out, indent=1))
