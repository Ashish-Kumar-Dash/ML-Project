"""
Temporal-locality audit across all nine time-mixing layers.

For each layer, measures how much of its weight mass sits near the scaled diagonal
of each source block, and compares that against (a) the same layer at random
initialisation and (b) the chance level a uniform matrix would give.

locality ratio = diag_mass / chance
  ~1.0  -> no temporal locality learned
  >1.3  -> the layer concentrated mass near the diagonal; a small shared kernel
           would capture the same thing far more cheaply

Block order is taken from get_complete_data (model.py:281-338), not from the
layer names -- v_alp is named (a,l) but concatenates l then a.

Usage:
  python tools/diag_mass_all.py --ckpt runs/mosi_bs64_e200/checkpoint.pt
  python tools/diag_mass_all.py --ckpt ... --lengths 50,375,500   # unaligned
"""

import argparse
import sys
import types
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

try:
    from src.model import PromptModel  # noqa: E402
except ModuleNotFoundError:
    sys.path.insert(0, str(REPO_ROOT / "third_party" / "MPLMM"))
    from src.model import PromptModel  # noqa: E402

PROMPT_LENGTH = 16
PROMPT_DIM = 30
BAND = 3  # positions either side of the scaled diagonal

# layer -> (output modality, [source modalities IN CONCATENATION ORDER])
# verified against get_complete_data, model.py:281-338
LAYERS = {
    "l_avp": ("l", ["a", "v"]),
    "a_lvp": ("a", ["l", "v"]),
    "v_alp": ("v", ["l", "a"]),
    "l_ap":  ("l", ["a"]),
    "l_vp":  ("l", ["v"]),
    "a_lp":  ("a", ["l"]),
    "a_vp":  ("a", ["v"]),
    "v_ap":  ("v", ["a"]),
    "v_lp":  ("v", ["l"]),
}


def make_hp(seq_len, dims=(300, 5, 20)):
    h = types.SimpleNamespace()
    h.orig_d_l, h.orig_d_a, h.orig_d_v = dims
    h.seq_len = seq_len
    h.layers, h.num_heads = 5, 5
    for k in ("attn_dropout", "attn_dropout_a", "attn_dropout_v",
              "relu_dropout", "res_dropout", "out_dropout"):
        setattr(h, k, 0.1)
    h.embed_dropout = 0.25
    h.attn_mask = True
    h.output_dim = 1
    h.prompt_dim = PROMPT_DIM
    h.prompt_length = PROMPT_LENGTH
    h.proj_dim = 30
    return h


def diag_mass(block, band=BAND):
    """Share of |weight| within `band` of the block's scaled diagonal."""
    l_out, l_in = block.shape
    a = np.abs(block)
    if a.sum() == 0:
        return float("nan")
    rows = np.arange(l_out)[:, None]
    cols = np.arange(l_in)[None, :]
    centre = rows * (l_in / l_out)
    near = np.abs(cols - centre) <= band
    return float(a[near].sum() / a.sum())


def chance(l_in, band=BAND):
    """What a uniform matrix would give: (2*band+1) columns out of l_in."""
    return min(1.0, (2 * band + 1) / l_in)


def get_state(ckpt):
    obj = torch.load(ckpt, map_location="cpu", weights_only=False)
    if hasattr(obj, "state_dict"):
        return obj.state_dict()
    if isinstance(obj, dict):
        for k in ("state_dict", "model_state_dict"):
            if k in obj:
                return obj[k]
        return obj
    raise TypeError(f"cannot extract a state_dict from {type(obj)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--lengths", default="50,50,50",
                    help="l,a,v sequence lengths of the checkpoint")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    L, A, V = (int(x) for x in args.lengths.split(","))
    lens = {"l": L, "a": A, "v": V}

    sd = get_state(args.ckpt)
    torch.manual_seed(args.seed)
    ref = PromptModel(make_hp((L, A, V)))

    print("=" * 94)
    print("Temporal-locality audit — all nine time-mixing layers")
    print(f"checkpoint: {args.ckpt}")
    print(f"lengths: l={L} a={A} v={V}   band=±{BAND}")
    print("=" * 94)
    print(f"{'layer':<8}{'out':>5}{'src':>6}{'params':>10}"
          f"{'trained':>10}{'random':>9}{'chance':>9}{'ratio':>8}   verdict")
    print("-" * 94)

    ratios = []
    for name, (out_mod, in_mods) in LAYERS.items():
        key = next((k for k in sd if k.endswith(f"{name}.conv.weight")), None)
        if key is None:
            raise KeyError(
                f"{name}.conv.weight not in {args.ckpt} — "
                "is this a stage-two PromptModel checkpoint?")

        l_out = lens[out_mod]
        l_in = PROMPT_LENGTH + sum(lens[m] for m in in_mods)
        w = sd[key].detach().squeeze(-1).cpu().numpy()
        assert w.shape == (l_out, l_in), \
            f"{name}: checkpoint has {w.shape}, expected {(l_out, l_in)}"
        w_rand = getattr(ref, name).conv.weight.detach().squeeze(-1).numpy()

        off = PROMPT_LENGTH
        for m in in_mods:
            n = lens[m]
            d_tr = diag_mass(w[:, off:off + n])
            d_rd = diag_mass(w_rand[:, off:off + n])
            ch = chance(n)
            ratio = d_tr / ch
            ratios.append(ratio)
            verdict = ("no locality" if ratio < 1.15
                       else "WEAK locality" if ratio < 1.3
                       else "LOCALITY — investigate")
            print(f"{name:<8}{out_mod:>5}{m:>6}{w.size:>10,}"
                  f"{d_tr:>10.4f}{d_rd:>9.4f}{ch:>9.4f}{ratio:>8.2f}   {verdict}")
            off += n

    print("-" * 94)
    r = np.array(ratios)
    print(f"{'ALL':<8}{'':<11}{'':>10}{'':>10}{'':>9}{'':>9}"
          f"{r.mean():>8.2f}   mean over {len(r)} source blocks "
          f"(min {r.min():.2f}, max {r.max():.2f})")
    print()
    print("ratio = trained diag mass / chance. A trained layer that sits at ~1.0 has")
    print("learned no temporal locality: its weight is spread as evenly across input")
    print("positions as an untrained matrix. The 'random' column is the sanity check —")
    print("it should track 'chance' closely.")


if __name__ == "__main__":
    main()
