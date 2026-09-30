"""
Empirical receptive-field probe for the generation module's time-mixing layer.

Claim under test: because the time axis is moved into the convolution's channel
slot by a transpose, kernel_size=1 restricts the FEATURE axis, not time. The
layer is therefore dense over time (every output position draws on every input
position) and pointwise over features.

Method: perturb exactly one input position and count how many output positions
move. This measures the property rather than arguing it from the source.

Usage:  python tools/receptive_field.py
        python tools/receptive_field.py > results/receptive_field.txt
"""

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.model import MLPLayer  # noqa: E402

PROMPT_LENGTH = 16
PROMPT_DIM = 30
EPS = 1e-6
KICK = 5.0


def forward_as_model_does(layer, x):
    """Exactly the call in get_complete_data (model.py:286).

    x arrives as (batch, prompt_dim, time). The model transposes, applies the
    Conv1d, and transposes back -- so during the convolution the TIME axis
    occupies the channel slot.
    """
    return layer(x.transpose(1, 2)).transpose(1, 2)


def probe(l_out, in_lens, seed=0):
    l_in = PROMPT_LENGTH + sum(in_lens)
    torch.manual_seed(seed)
    layer = MLPLayer(l_in, l_out, True)

    x = torch.randn(1, PROMPT_DIM, l_in)
    y0 = forward_as_model_does(x=x, layer=layer)

    # --- perturb ONE input time position ------------------------------------
    t_idx = l_in // 2
    x_t = x.clone()
    x_t[0, :, t_idx] += KICK
    y_t = forward_as_model_does(x=x_t, layer=layer)
    moved_time = int(((y_t - y0).abs().sum(dim=1)[0] > EPS).sum())

    # --- perturb ONE input feature channel -----------------------------------
    f_idx = PROMPT_DIM // 2
    x_f = x.clone()
    x_f[0, f_idx, :] += KICK
    y_f = forward_as_model_does(x=x_f, layer=layer)
    moved_feat = int(((y_f - y0).abs().sum(dim=2)[0] > EPS).sum())

    return l_in, moved_time, moved_feat


def main():
    cases = [
        ("l_avp, MOSI aligned   (l=50, a=50,  v=50)",  50, (50, 50)),
        ("l_avp, MOSI unaligned (l=50, a=375, v=500)", 50, (375, 500)),
        ("a_lvp, MOSI unaligned (a=375)",             375, (50, 500)),
    ]

    print("=" * 72)
    print("Receptive-field probe -- MPLMM time-mixing layer (MLPLayer, is_Fusion=True)")
    print("repo: github.com/zrguo/MPLMM   file: src/model.py:466-476")
    print("=" * 72)
    print()

    all_dense, all_pointwise = True, True
    for label, l_out, in_lens in cases:
        l_in, moved_time, moved_feat = probe(l_out, in_lens)
        dense = (moved_time == l_out)
        pointwise = (moved_feat == 1)
        all_dense &= dense
        all_pointwise &= pointwise

        print(label)
        print(f"  input  (prompt_dim={PROMPT_DIM}, time={l_in})"
              f"   ->   output (prompt_dim={PROMPT_DIM}, time={l_out})")
        print(f"  perturb 1 input TIME position    -> {moved_time:>4} of {l_out:>4} "
              f"output time positions move   {'[dense over time]' if dense else '[NOT dense]'}")
        print(f"  perturb 1 input FEATURE channel  -> {moved_feat:>4} of "
              f"{PROMPT_DIM:>4} output feature indices move   "
              f"{'[pointwise over features]' if pointwise else '[NOT pointwise]'}")
        print()

    print("=" * 72)
    assert all_dense, "layer is not dense over time -- claim falsified"
    assert all_pointwise, "layer is not pointwise over features -- claim falsified"
    print("Result: the layer mixes time globally and features not at all.")
    print("kernel_size=1 constrains the FEATURE axis, because the transpose put")
    print("time in the convolution's channel slot. There is no weight sharing")
    print("across time and no translation-equivariance: a shifted input produces")
    print("an unrelated output, and the weight count is bound to the exact")
    print("sequence length the layer was instantiated at.")


if __name__ == "__main__":
    main()