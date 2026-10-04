#!/usr/bin/env python3
"""Pooled length-ladder resampler for unaligned CMU-MOSI (MulT GloVe-300 pickles).

Builds one pickle per rung l of the ladder. Text (T=50) is carried through
untouched; audio (T=375) and vision (T=500) are resampled along time with
torch.nn.functional.adaptive_avg_pool1d.

Why adaptive pooling and not a reshape-based pool: the ratios are non-integer
(375/50 = 7.5, 375/100 = 3.75, 500/200 = 2.5), so x.reshape(N, l, -1, D).mean(2)
either raises or silently drops the trailing remainder steps. Adaptive pooling
splits T into l contiguous bins with
    start_i = floor(i * T / l),  end_i = ceil((i + 1) * T / l)
so every input step lands in at least one bin; where the ratio is non-integer,
boundary steps are shared by two adjacent bins. Both properties are measured
and written to the report.

Native-rung identity check: at l = T every bin has width 1, so pooling must
reproduce the input. The native rung is produced by the same pool_time() path
as the other rungs (not by copying the source), and the tool asserts
|pooled - src| <= atol + rtol*|src| elementwise for every split/modality,
recording max abs/rel error and whether the result is bit-exact.

Pooling is UNMASKED. The pickles carry no audio_lengths / vision_lengths and
the sequences are front-padded with zero rows, so early bins average padding.
That limitation is recorded in docs/DEVIATIONS.md ("Sequence padding without
length fields"); this tool does not attempt to correct for it.

Usage:
    python3 tools/pool_resample.py \
        --src   ~/Downloads/ML-project/ML-Project/data/mosi_data_noalign.pkl \
        --out   ~/Downloads/ML-project/ML-Project/data/ladder \
        --report ~/Downloads/ML-project/mplmm-length/results
"""

import argparse
import hashlib
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SPLITS = ("train", "valid", "test")
POOLED = ("audio", "vision")          # text (T=50) is the reference length, left as-is
CARRY = ("text", "id", "labels")


def bin_edges(T, l):
    """Bin boundaries used by adaptive_avg_pool1d."""
    return [(int(math.floor(i * T / l)), int(math.ceil((i + 1) * T / l))) for i in range(l)]


def bin_stats(T, l):
    edges = bin_edges(T, l)
    widths = np.array([e - s for s, e in edges])
    cover = np.zeros(T, dtype=np.int64)
    for s, e in edges:
        cover[s:e] += 1
    return dict(
        T=int(T), l=int(l),
        ratio=round(T / l, 6),
        width_min=int(widths.min()), width_max=int(widths.max()),
        steps_dropped=int((cover == 0).sum()),
        steps_shared=int((cover > 1).sum()),
    )


def pool_time(x, l, chunk=512):
    """(N, T, D) -> (N, l, D), averaging over time. Preserves dtype."""
    dtype = x.dtype
    out = np.empty((x.shape[0], l, x.shape[2]), dtype=dtype)
    for i in range(0, x.shape[0], chunk):
        t = torch.from_numpy(np.ascontiguousarray(x[i:i + chunk]))
        t = t.permute(0, 2, 1)                       # (n, D, T)
        t = F.adaptive_avg_pool1d(t, l)              # (n, D, l)
        out[i:i + chunk] = t.permute(0, 2, 1).numpy().astype(dtype, copy=False)
    return out


def identity_check(src, pooled, atol, rtol, where):
    """Assert pooled ~= src elementwise; return error stats."""
    assert pooled.shape == src.shape, f"{where}: shape {pooled.shape} != {src.shape}"
    assert pooled.dtype == src.dtype, f"{where}: dtype {pooled.dtype} != {src.dtype}"
    s64, p64 = src.astype(np.float64), pooled.astype(np.float64)
    err = np.abs(p64 - s64)
    tol = atol + rtol * np.abs(s64)
    n_bad = int((err > tol).sum())
    stats = dict(
        max_abs_err=float(err.max()),
        max_rel_err=float((err / np.maximum(np.abs(s64), 1e-12)).max()),
        n_outside_tol=n_bad,
        bit_exact=bool(np.array_equal(pooled, src)),
        atol=atol, rtol=rtol,
    )
    assert n_bad == 0, f"{where}: native-length pooling is not near-identity: {stats}"
    return stats


def zero_row_frac(x):
    return float(1.0 - (np.abs(x).sum(-1) > 0).mean())


def sha256(path, buf=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(buf), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--report", required=True, type=Path)
    ap.add_argument("--rungs", default="50,100,200,native",
                    help="comma-separated target lengths; 'native' = keep source T")
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--identity-atol", type=float, default=1e-6)
    ap.add_argument("--identity-rtol", type=float, default=1e-5)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    args.report.mkdir(parents=True, exist_ok=True)

    print(f"loading {args.src}", flush=True)
    data = pickle.load(open(args.src, "rb"))

    native = {m: int(np.asarray(data["train"][m]).shape[1]) for m in POOLED}
    nonfinite = {}
    for s in SPLITS:
        for m in POOLED + ("text",):
            a = np.asarray(data[s][m])
            nonfinite[f"{s}/{m}"] = int((~np.isfinite(a)).sum())
    if any(nonfinite.values()):
        sys.exit(f"non-finite values present, refusing to average: {nonfinite}")

    report = {
        "source": str(args.src),
        "source_sha256": sha256(args.src),
        "native_T": native,
        "text_T": int(np.asarray(data["train"]["text"]).shape[1]),
        "nonfinite_counts": nonfinite,
        "pooling": "torch.nn.functional.adaptive_avg_pool1d, unmasked",
        "torch": torch.__version__,
        "rungs": {},
    }

    for rung in [r.strip() for r in args.rungs.split(",")]:
        tgt = {m: native[m] for m in POOLED} if rung == "native" else {m: int(rung) for m in POOLED}
        tag = "native" if rung == "native" else f"l{rung}"
        name = f"{args.src.stem}_{tag}.pkl"
        dest = args.out / name
        print(f"rung {tag}: audio {native['audio']}->{tgt['audio']}, "
              f"vision {native['vision']}->{tgt['vision']}", flush=True)

        entry = {
            "file": str(dest),
            "targets": tgt,
            "bins": {m: bin_stats(native[m], tgt[m]) for m in POOLED},
            "splits": {},
        }

        new = {}
        for s in SPLITS:
            sp = data[s]
            ns = {k: sp[k] for k in CARRY if k in sp}
            info = {}
            for m in POOLED:
                a = np.asarray(sp[m])
                p = pool_time(a, tgt[m], args.chunk)
                ident = None
                if tgt[m] == a.shape[1]:
                    ident = identity_check(a, p, args.identity_atol, args.identity_rtol, f"{s}/{m}")
                ns[m] = p
                info[m] = dict(
                    shape=list(p.shape), dtype=str(p.dtype),
                    zero_row_frac_src=round(zero_row_frac(a), 4),
                    zero_row_frac_pooled=round(zero_row_frac(p), 4),
                    abs_mean_src=float(np.abs(a).mean()),
                    abs_mean_pooled=float(np.abs(p).mean()),
                )
                if ident is not None:
                    info[m]["identity"] = ident
            t = np.asarray(sp["text"])
            info["text"] = dict(shape=list(t.shape), dtype=str(t.dtype), pooled=False)
            info["n"] = int(np.asarray(sp["labels"]).shape[0])
            entry["splits"][s] = info
            new[s] = ns

        with open(dest, "wb") as fh:
            pickle.dump(new, fh, protocol=4)
        entry["size_bytes"] = dest.stat().st_size
        entry["sha256"] = sha256(dest)
        report["rungs"][tag] = entry
        del new

    (args.report / "pool_ladder.json").write_text(json.dumps(report, indent=1))
    sums = "".join(f"{e['sha256']}  {Path(e['file']).name}\n"
                   for _, e in sorted(report["rungs"].items(), key=lambda kv: Path(kv[1]["file"]).name))
    (args.out / "SHA256SUMS").write_text(sums)

    lines = []
    lines.append("Pooled length ladder - unaligned CMU-MOSI (MulT GloVe-300)")
    lines.append(f"source     : {report['source']}")
    lines.append(f"sha256     : {report['source_sha256']}")
    lines.append(f"resampler  : {report['pooling']} (torch {report['torch']})")
    lines.append(f"text       : T={report['text_T']}, not pooled")
    lines.append("")
    lines.append("Bin geometry (adaptive_avg_pool1d; dropped = input steps in no bin,")
    lines.append("shared = input steps counted by two adjacent bins)")
    lines.append(f"{'rung':>8} {'mod':>7} {'T':>5} {'->':>2} {'l':>5} {'ratio':>8} "
                 f"{'width':>7} {'dropped':>8} {'shared':>7}")
    for tag, e in report["rungs"].items():
        for m in POOLED:
            b = e["bins"][m]
            lines.append(f"{tag:>8} {m:>7} {b['T']:>5} {'->':>2} {b['l']:>5} {b['ratio']:>8.4f} "
                         f"{str(b['width_min']) + '-' + str(b['width_max']):>7} "
                         f"{b['steps_dropped']:>8} {b['steps_shared']:>7}")
    lines.append("")
    lines.append("Files and shapes (train/valid/test)")
    for tag, e in report["rungs"].items():
        lines.append(f"  {Path(e['file']).name}  ({e['size_bytes'] / 1e6:.1f} MB)")
        lines.append(f"    sha256 {e['sha256']}")
        for m in ("text",) + POOLED:
            shapes = " ".join("x".join(str(v) for v in e["splits"][s][m]["shape"]) for s in SPLITS)
            lines.append(f"    {m:<7} {shapes}")
    lines.append("")
    lines.append("All-zero-timestep fraction, source -> pooled (train/valid/test)")
    for tag, e in report["rungs"].items():
        for m in POOLED:
            cells = "  ".join(
                f"{e['splits'][s][m]['zero_row_frac_src']:.3f}->{e['splits'][s][m]['zero_row_frac_pooled']:.3f}"
                for s in SPLITS)
            lines.append(f"  {tag:>8} {m:>7}  {cells}")
    if "native" in report["rungs"]:
        lines.append("")
        lines.append("Native-length identity check (pool_time at l = T vs source; asserted)")
        lines.append(f"  {'split':>6} {'mod':>7} {'max_abs_err':>12} {'max_rel_err':>12} "
                     f"{'outside_tol':>11} {'bit_exact':>9}")
        e = report["rungs"]["native"]
        for s in SPLITS:
            for m in POOLED:
                d = e["splits"][s][m]["identity"]
                lines.append(f"  {s:>6} {m:>7} {d['max_abs_err']:>12.3e} {d['max_rel_err']:>12.3e} "
                             f"{d['n_outside_tol']:>11} {str(d['bit_exact']):>9}")
        d = e["splits"]["train"]["audio"]["identity"]
        lines.append(f"  tolerance: |pooled - src| <= {d['atol']:g} + {d['rtol']:g}*|src|")
    lines.append("")
    lines.append("Pooling is unmasked: sequences are front-padded with zero rows and the")
    lines.append("pickles carry no length fields, so the leading bins average padding.")
    lines.append("See docs/DEVIATIONS.md, 'Sequence padding without length fields'.")
    txt = "\n".join(lines) + "\n"
    (args.report / "pool_ladder.txt").write_text(txt)
    print(txt, flush=True)


if __name__ == "__main__":
    main()
