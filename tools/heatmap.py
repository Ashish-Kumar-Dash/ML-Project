"""
Weight heatmap for the generation module's time-mixing layer.

Renders l_avp's learned weight as an image: output time positions on the y
axis, input positions (generative prompt | audio | visual) on the x axis.

What to look for:
  - no diagonal structure  -> the layer learned an arbitrary position-to-
    position remap. 10^6 parameters spent on something with no temporal
    structure at all; the case for replacing it gets stronger.
  - visible diagonal band  -> a small shared kernel would capture the same
    thing at a fraction of the cost. An even sharper result.

Run it today on random weights to debug the plotting. When the trained
checkpoint exists, pass --ckpt and nothing else changes.

Usage:
  python tools/heatmap.py                                   # random init, both shapes
  python tools/heatmap.py --ckpt results/l1_seed1.pt        # trained weights
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import torch                      # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.model import MLPLayer  # noqa: E402

PROMPT_LENGTH = 16
FIG_DIR = REPO_ROOT / "figures"

# Ink colours that hold up in both light and dark rendering.
INK = "#33322E"
MUTED = "#6F6E69"
RULE = "#B2AFA6"


def get_weight(shape_name, l_out, in_lens, ckpt=None, seed=0):
    """Return l_avp's weight as (l_out, l_in), from a checkpoint or random init."""
    l_in = PROMPT_LENGTH + sum(in_lens)

    if ckpt is not None:
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        if hasattr(sd, "state_dict"):
            sd = sd.state_dict()
        key = next((k for k in sd if k.endswith("l_avp.conv.weight")), None)
        if key is None:
            raise KeyError(
                f"no l_avp.conv.weight in {ckpt}. Keys look like: "
                f"{list(sd)[:5]}"
            )
        w = sd[key]
        source = f"trained ({Path(ckpt).name})"
    else:
        torch.manual_seed(seed)
        w = MLPLayer(l_in, l_out, True).conv.weight
        source = "randomly initialised"

    w = w.detach().squeeze(-1).numpy()   # (l_out, l_in, 1) -> (l_out, l_in)
    return w, source


def plot(w, source, shape_name, in_lens, out_path):
    l_out, l_in = w.shape
    a_len, v_len = in_lens
    lim = float(np.abs(w).max())

    fig_w = 10 if l_in > 300 else 6.4
    fig, ax = plt.subplots(figsize=(fig_w, 3.6), dpi=160)

    im = ax.imshow(w, cmap="RdBu_r", vmin=-lim, vmax=lim,
                   aspect="auto", interpolation="nearest")

    # segment dividers: generative prompt | audio | visual
    for x in (PROMPT_LENGTH - 0.5, PROMPT_LENGTH + a_len - 0.5):
        ax.axvline(x, color=INK, lw=1.0, alpha=0.65)

    seg_mid = [PROMPT_LENGTH / 2,
               PROMPT_LENGTH + a_len / 2,
               PROMPT_LENGTH + a_len + v_len / 2]
    ax.set_xticks(seg_mid)
    ax.set_xticklabels([f"prompt\n({PROMPT_LENGTH})",
                        f"audio\n({a_len})",
                        f"visual\n({v_len})"], color=MUTED, fontsize=9)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", colors=MUTED, labelsize=9)

    ax.set_ylabel("output time position", color=MUTED, fontsize=9)
    ax.set_title(
        f"l_avp weight — MOSI {shape_name}  ({l_out} x {l_in} = {w.size:,} params, {source})",
        color=INK, fontsize=10.5, pad=10)

    for s in ax.spines.values():
        s.set_color(RULE)
        s.set_linewidth(0.8)

    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015)
    cb.outline.set_color(RULE)
    cb.ax.tick_params(colors=MUTED, labelsize=8)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # A crude numeric companion to the eye test: how much of the total weight
    # mass sits near the diagonal of the audio block? Reported, not asserted --
    # it is a descriptive statistic, not a claim.
    band = diagonal_mass(w, a_len)
    return band


def diagonal_mass(w, a_len, width=3):
    """Fraction of |weight| in the audio block lying within `width` of its
    scaled diagonal. Uniform noise gives roughly (2*width+1)/a_len."""
    l_out = w.shape[0]
    blk = np.abs(w[:, PROMPT_LENGTH:PROMPT_LENGTH + a_len])
    if blk.sum() == 0:
        return float("nan")
    rows = np.arange(l_out)[:, None]
    cols = np.arange(a_len)[None, :]
    centre = rows * (a_len / l_out)
    near = np.abs(cols - centre) <= width
    return float(blk[near].sum() / blk.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None,
                    help="trained checkpoint or state_dict; omit for random init")
    args = ap.parse_args()

    FIG_DIR.mkdir(exist_ok=True)
    shapes = [
        ("aligned",   50, (50, 50)),
        ("unaligned", 50, (375, 500)),
    ]

    print(f"{'shape':<12}{'params':>12}{'diag mass':>12}{'chance':>10}   file")
    for name, l_out, in_lens in shapes:
        w, source = get_weight(name, l_out, in_lens, ckpt=args.ckpt)
        out = FIG_DIR / f"l_avp_{name}{'_trained' if args.ckpt else '_random'}.png"
        band = plot(w, source, name, in_lens, out)
        chance = 7 / in_lens[0]
        print(f"{name:<12}{w.size:>12,}{band:>12.3f}{chance:>10.3f}   {out.name}")

    print()
    print("diag mass = share of |weight| in the audio block within 3 positions of")
    print("its scaled diagonal; 'chance' is what uniform noise would give. On a")
    print("randomly initialised layer the two should match. A trained layer that")
    print("still matches has learned no temporal locality.")


if __name__ == "__main__":
    main()