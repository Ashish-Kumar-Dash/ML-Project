
"""L2'-CLIP scale fix: divide the PLS-20 vision features by ONE scalar (fit on train real rows) so their median row norm
equals L1 Facet's. Relative variance across PLS components is preserved; padding rows stay exactly zero."""
import pickle, json, hashlib, numpy as np
src, dst = "data/clip/mosi_l2pclip.pkl", "data/clip/mosi_l2pclipsc.pkl"
d = pickle.load(open(src, "rb")); ref = pickle.load(open("data/mosi_data.pkl", "rb"), encoding="latin1")
def med_norm(X):
    X = np.asarray(X, dtype=np.float64); m = np.abs(X).sum(-1) > 0; return float(np.median(np.linalg.norm(X[m], axis=1)))
target = med_norm(ref["train"]["vision"]); cur = med_norm(d["train"]["vision"]); s = cur / target
for k in ("train", "valid", "test"):
    d[k]["vision"] = (np.asarray(d[k]["vision"], dtype=np.float32) / np.float32(s)).astype(np.float32)
    assert [r[0] for r in d[k]["id"]] == [r[0] for r in ref[k]["id"]]
pickle.dump(d, open(dst, "wb"), protocol=4)
meta = json.load(open(src + ".meta.json")); meta["vision"]["rescale"] = {"divisor": s, "fit_on": "train real rows", "target_median_row_norm_L1_facet": target,
    "before_median_row_norm": cur, "after_median_row_norm": med_norm(d["train"]["vision"]), "reason": "PLS scores ~13x Facet scale caused majority-class collapse in all 3 dense seeds"}
json.dump(meta, open(dst + ".meta.json", "w"), indent=1)
h = hashlib.sha256(open(dst, "rb").read()).hexdigest()
open("data/clip/SHA256SUMS", "a").write(f"{h}  {dst}\n")
print(json.dumps(meta["vision"]["rescale"], indent=1), h)
