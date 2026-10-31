
"""Build real-CLIP rung pickles (SOP items 2+5): L2-CLIP (512-d) and L2'-CLIP (PLS->20-d), layout-matched to L1.
Vision rows are placed in the text's word slots (the slots L1's word-aligned Facet occupies); each clip's native
30-fps CLIP frames are split into k contiguous windows (k = number of word slots) by adaptive average pooling.
Text (GloVe-300) and audio (COVAREP-5) are copied unchanged from data/mosi_data.pkl."""
import sys, json, pickle, hashlib, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
sys.path.insert(0, ".")
from sklearn.cross_decomposition import PLSRegression
from tools.merge_multimodal_features import merge_datasets
REF, NAT, OUT = Path("data/mosi_data.pkl"), Path("data/mosi_clip_b32_native.pkl"), Path("data/clip")
OUT.mkdir(exist_ok=True)
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
ref = pickle.load(open(REF, "rb"), encoding="latin1"); nat = pickle.load(open(NAT, "rb"))
S = ("train", "valid", "test")
def word_slots(s): return np.abs(np.asarray(ref[s]["text"])).sum(-1) > 0
vis512, stats = {}, {}
for s in S:
    ids = [r[0].decode() for r in ref[s]["id"]]; assert nat["ids"][s] == ids
    slots = word_slots(s); N, L = slots.shape; V = np.zeros((N, L, 512), np.float32); k0 = 0
    for i, fr in enumerate(nat["frames"][s]):
        pos = np.flatnonzero(slots[i]); k = len(pos)
        if k == 0: k0 += 1; continue
        x = torch.from_numpy(fr.astype(np.float32)).T.unsqueeze(0)          # (1, 512, n_frames)
        V[i, pos] = F.adaptive_avg_pool1d(x, k).squeeze(0).T.numpy()
    vis512[s] = V
    l1v = np.abs(np.asarray(ref[s]["vision"])).sum(-1) > 0
    stats[s] = {"clips": N, "clips_without_word_slots": k0, "slot_agreement_with_L1_facet_rows": float((l1v == slots).mean())}
# PLS 512 -> 20, fitted on TRAIN clip-level means only (label-aware, as tools/fit_l2_projection.py does for L2')
def clip_mean(V, m): return (V * m[..., None]).sum(1) / np.maximum(m.sum(1, keepdims=True), 1)
ytr = np.asarray(ref["train"]["labels"]).reshape(len(ref["train"]["labels"]), -1)[:, 0]
pls = PLSRegression(n_components=20).fit(clip_mean(vis512["train"], word_slots("train")), ytr)
vis20 = {}
for s in S:
    m = word_slots(s); N, L, _ = vis512[s].shape
    Z = pls.transform(vis512[s].reshape(N * L, 512)).reshape(N, L, 20).astype(np.float32)
    Z[~m] = 0.0                                                             # padding stays exactly zero
    vis20[s] = Z
meta_common = {"synthetic": False, "model": "ViT-B/32", "source_native": str(NAT), "source_native_sha256": sha(NAT),
               "layout": "text word slots (L1 alignment structure); per-clip uniform split of 30-fps frames into k windows",
               "audio": "COVAREP 5-d from data/mosi_data.pkl (unchanged)", "text": "GloVe 300-d from data/mosi_data.pkl (unchanged)",
               "pipeline_mix": "CLIP vision + COVAREP audio + GloVe text", "slot_stats": stats}
built = {}
for tag, vis, extra in (("l2clip", vis512, {"vision_dim": 512}), ("l2pclip", vis20, {"vision_dim": 20, "projection": "PLSRegression(n_components=20) fit on train clip-level means vs label"})):
    vp = OUT / f"_vision_{tag}.pkl"
    pickle.dump({**vis, "ids": {s: [r[0].decode() for r in ref[s]["id"]] for s in S}, "meta": {**meta_common, **extra}}, open(vp, "wb"), protocol=4)
    out = OUT / f"mosi_{tag}.pkl"
    merge_datasets(reference_pkl=REF, vision_pkl=vp, output_path=out, verify_dataloader=True)
    vp.unlink()
    built[tag] = {"path": str(out), "sha256": sha(out)}
with open(OUT / "SHA256SUMS", "w") as f:
    for p in (NAT, Path("data/mosi_clip_b32_v50_vision.pkl"), OUT / "mosi_l2clip.pkl", OUT / "mosi_l2pclip.pkl"):
        f.write(f"{sha(p)}  {p}\n")
print(json.dumps({"built": built, "slot_stats": stats}, indent=1))
print("BUILD_OK")
