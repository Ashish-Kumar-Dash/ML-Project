
"""L2'-CLIP-PCA: label-FREE dimension-matched control. PCA(512->20) fit on train clip-level means (no labels),
applied per row; padding rows stay zero; one scalar rescale so the median row norm equals L1 Facet's."""
import pickle, json, hashlib, numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
src = "data/clip/mosi_l2clip.pkl"; dst = "data/clip/mosi_l2pcaclip.pkl"
d = pickle.load(open(src, "rb")); ref = pickle.load(open("data/mosi_data.pkl", "rb"), encoding="latin1")
S = ("train", "valid", "test")
mask = {s: np.abs(np.asarray(d[s]["text"])).sum(-1) > 0 for s in S}
def cmean(V, m): return (V * m[..., None]).sum(1) / np.maximum(m.sum(1, keepdims=True), 1)
V = {s: np.asarray(d[s]["vision"], dtype=np.float64) for s in S}
pca = PCA(n_components=20, random_state=0).fit(cmean(V["train"], mask["train"]))
Z = {}
for s in S:
    N, L, D = V[s].shape; z = pca.transform(V[s].reshape(N * L, D)).reshape(N, L, 20); z[~mask[s]] = 0.0; Z[s] = z
med = lambda X, m: float(np.median(np.linalg.norm(X[m], axis=1)))
fm = {s: np.abs(np.asarray(ref[s]["vision"])).sum(-1) > 0 for s in S}
target = med(np.asarray(ref["train"]["vision"], dtype=np.float64), fm["train"]); div = med(Z["train"], mask["train"]) / target
for s in S: d[s]["vision"] = (Z[s] / div).astype(np.float32)
pickle.dump(d, open(dst, "wb"), protocol=4)
y = {s: np.asarray(ref[s]["labels"]).reshape(len(ref[s]["labels"]), -1)[:, 0] for s in S}
C = {s: cmean(d[s]["vision"].astype(np.float64), mask[s]) for s in S}
lr = LinearRegression().fit(C["train"], y["train"])
chk = {s: round(float(pearsonr(lr.predict(C[s]), y[s])[0]), 3) for s in S}
meta = json.load(open(src + ".meta.json")); meta["vision"].update({"vision_dim": 20, "projection": "PCA(n_components=20) fit on TRAIN clip-level means, label-free",
    "pca_explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()), "rescale_divisor": div, "target_median_row_norm_L1_facet": target,
    "linear_fit_r_train_valid_test": chk})
json.dump(meta, open(dst + ".meta.json", "w"), indent=1)
h = hashlib.sha256(open(dst, "rb").read()).hexdigest(); open("data/clip/SHA256SUMS", "a").write(f"{h}  {dst}\n")
print(json.dumps({"evr_sum": meta["vision"]["pca_explained_variance_ratio_sum"], "divisor": div, "linear_fit_r": chk, "sha": h}))
