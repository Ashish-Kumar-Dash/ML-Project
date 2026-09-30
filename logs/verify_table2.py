
import pickle, gc, json, numpy as np
EXP = {
 "mosi_data.pkl":         dict(n=(1284,229,686),   text=(50,300), audio=(50,5),  vision=(50,20)),
 "mosi_data_noalign.pkl": dict(n=(1284,229,686),   text=(50,300), audio=(375,5), vision=(500,20)),
 "mosei_senti_data.pkl":  dict(n=(16265,1869,4643),text=(50,300), audio=(50,74), vision=(50,35)),
}
out = {}
for f, e in EXP.items():
    d = pickle.load(open("data/"+f, "rb")); R = {}
    for s, n in zip(("train","valid","test"), e["n"]):
        sp = d[s]; r = {"keys": sorted(sp.keys()), "len_keys": [k for k in sp if "len" in k.lower()]}
        chk = {"N": [int(sp["labels"].shape[0]), n, int(sp["labels"].shape[0]) == n]}
        for m in ("text","audio","vision"):
            a = np.asarray(sp[m]); want = (n,)+e[m]; chk[m] = [list(a.shape), list(want), tuple(a.shape) == want]
            b = np.where(np.isfinite(a), a, 0.0)
            live = np.abs(b).sum(-1) > 0
            T = a.shape[1]; L = live.sum(1); idx = np.arange(T)[None]
            r[m] = dict(dtype=str(a.dtype), nan=int(np.isnan(a).sum()), neginf=int(np.isneginf(a).sum()),
                        posinf=int(np.isposinf(a).sum()),
                        pad_frac=round(float(1 - live.mean()), 4), len_min=int(L.min()), len_med=float(np.median(L)),
                        len_max=int(L.max()), n_full=int((L == T).sum()), n_empty=int((L == 0).sum()),
                        left_pad=bool(np.all(live == (idx >= T - L[:, None]))),
                        right_pad=bool(np.all(live == (idx < L[:, None]))),
                        interior_zero_rows=int(((~live) & (idx >= (T - L)[:, None])).sum()))
        r["checks"] = chk
        lab = np.asarray(sp["labels"]); r["labels"] = dict(shape=list(lab.shape), min=float(lab.min()), max=float(lab.max()))
        R[s] = r
    out[f] = R; del d; gc.collect()
json.dump(out, open("logs/verify_table2.json", "w"), indent=1)

