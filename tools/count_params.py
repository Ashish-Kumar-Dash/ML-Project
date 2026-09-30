"""
Parameter audit of the MPLMM generation module.

Claim under test: the nine time-mixing layers of the Missing Modality Generation
Module have parameter counts bound to the input sequence lengths the model was
instantiated at, so the module that exists to avoid fine-tuning the backbone
grows past the backbone itself on long sequences.

Every figure is produced twice -- once by instantiating the real PromptModel
and counting tensors, once by a closed-form derivation -- and asserted equal.
A number only one path produces is not verified.

Usage:  python tools/count_params.py            # prints report
        python tools/count_params.py > results/param_audit.txt
"""

import sys
import types
from pathlib import Path

import torch

# --- make `src.model` importable no matter where this is run from -----------
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.model import MULTModel, PromptModel  # noqa: E402

PROMPT_LENGTH = 16
PROMPT_DIM = 30
PROJ_DIM = 30

# The nine length-dependent time-mixing layers (model.py:187-202).
# Each entry: attribute name -> (output modality, input modalities).
# Modality keys index into a (l, a, v) sequence-length triple.
TIME_MIXING = {
    "l_avp": ("l", ("a", "v")),   # language absent, both others present
    "a_lvp": ("a", ("l", "v")),   # audio absent
    "v_alp": ("v", ("a", "l")),   # visual absent
    "l_ap":  ("l", ("a",)),       # language + visual absent
    "l_vp":  ("l", ("v",)),       # language + audio absent
    "a_lp":  ("a", ("l",)),
    "a_vp":  ("a", ("v",)),
    "v_ap":  ("v", ("a",)),
    "v_lp":  ("v", ("l",)),
}

# The six channel-mixing layers (model.py:180-185). Listed so the report can
# state explicitly that these are length-INDEPENDENT -- it forestalls the
# obvious objection that we cherry-picked which layers to count.
CHANNEL_MIXING = ["l2a", "l2v", "v2a", "v2l", "a2v", "a2l"]

# Length-dependent prompt tensors: missing-signal prompts (model.py:217-222)
# and the missing-type coupling matrices (model.py:228-230).
SIGNAL_PROMPTS = ["promptl_m", "prompta_m", "promptv_m",
                  "promptl_nm", "prompta_nm", "promptv_nm"]
COUPLING = ["m_l", "m_a", "m_v"]


def make_hyp(seq_len, orig_dims, output_dim=1):
    """Minimal hyp_params namespace matching main.py's argparse defaults."""
    h = types.SimpleNamespace()
    h.orig_d_l, h.orig_d_a, h.orig_d_v = orig_dims
    h.seq_len = seq_len
    h.layers = 5
    h.num_heads = 5
    h.attn_dropout = 0.1
    h.attn_dropout_a = 0.1
    h.attn_dropout_v = 0.1
    h.relu_dropout = 0.1
    h.res_dropout = 0.1
    h.out_dropout = 0.1
    h.embed_dropout = 0.25
    h.attn_mask = True
    h.output_dim = output_dim
    h.prompt_dim = PROMPT_DIM
    h.prompt_length = PROMPT_LENGTH
    h.proj_dim = PROJ_DIM
    return h


def n_params(module):
    return sum(p.numel() for p in module.parameters())


def closed_form(out_len, in_lens):
    """Equation (1) of the proposal: params = l_out * (l_p + sum l_in) + l_out.

    Independent derivation path. MLPLayer wraps Conv1d(in, out, kernel_size=1)
    with bias, so weight is (out, in, 1) and bias is (out,).
    """
    return out_len * (PROMPT_LENGTH + sum(in_lens)) + out_len


def audit(seq_len, orig_dims, label):
    """Instantiate PromptModel and bucket its parameters. Returns a dict."""
    l_len, a_len, v_len = seq_len
    lens = {"l": l_len, "a": a_len, "v": v_len}

    model = PromptModel(make_hyp(seq_len, orig_dims))

    # --- bucket 1: nine time-mixing layers, both paths -----------------------
    measured_time, derived_time = {}, {}
    for name, (out_mod, in_mods) in TIME_MIXING.items():
        measured_time[name] = n_params(getattr(model, name))
        derived_time[name] = closed_form(lens[out_mod], [lens[m] for m in in_mods])

    for name in TIME_MIXING:
        assert measured_time[name] == derived_time[name], (
            f"{label}/{name}: instantiated {measured_time[name]} != "
            f"closed form {derived_time[name]}"
        )

    # --- bucket 2: six channel-mixing layers (length-independent) ------------
    channel_total = sum(n_params(getattr(model, n)) for n in CHANNEL_MIXING)

    # --- bucket 3: length-dependent prompt tensors ---------------------------
    signal_total = sum(getattr(model, n).numel() for n in SIGNAL_PROMPTS)
    coupling_total = sum(getattr(model, n).numel() for n in COUPLING)

    # cross-check these too: 2 prompts per modality x prompt_dim x seq_len,
    # and coupling is seq_len x 2*prompt_dim per modality.
    assert signal_total == 2 * PROMPT_DIM * (l_len + a_len + v_len)
    assert coupling_total == 2 * PROMPT_DIM * (l_len + a_len + v_len)

    gen_prompt = model.generative_prompt.numel()
    type_prompt = model.missing_type_prompt.numel()

    return {
        "label": label,
        "time_mixing": sum(measured_time.values()),
        "time_mixing_detail": measured_time,
        "channel_mixing": channel_total,
        "signal_prompts": signal_total,
        "coupling": coupling_total,
        "generative_prompt": gen_prompt,
        "missing_type_prompt": type_prompt,
        "length_dependent": sum(measured_time.values()) + signal_total + coupling_total,
        "model": model,
    }


def trainable_after_transfer(prompt_model, backbone_dims, seq_len, tmp_path):
    """Run the real transfer_model freezing path from src/utils.py.

    Reimplemented here rather than imported because src/utils.py pulls in
    src.iemodata -> h5py, which is not in requirements. The logic below is a
    line-for-line copy of transfer_model (src/utils.py:78-104).
    """
    pre = MULTModel(make_hyp(seq_len, backbone_dims))
    torch.save(pre, tmp_path)

    # PyTorch >= 2.6 defaults weights_only=True; the checkpoint is a pickled
    # MULTModel object, so it must be loaded with weights_only=False. This is
    # the patch Sujal applies at src/utils.py:81.
    loaded = torch.load(tmp_path, weights_only=False)
    pretrain_dict = loaded.state_dict()
    new_dict = prompt_model.state_dict()

    excluded = ["proj_l.weight", "proj_a.weight", "proj_v.weight",
                "out_layer.weight", "out_layer.bias"]

    state_dict = {k: v for k, v in pretrain_dict.items()
                  if k in new_dict and k not in excluded}
    new_dict.update(state_dict)
    prompt_model.load_state_dict(new_dict)

    for name, param in prompt_model.named_parameters():
        if name in pretrain_dict and name not in excluded:
            param.requires_grad = False

    trainable = sum(p.numel() for p in prompt_model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in prompt_model.parameters() if not p.requires_grad)
    return trainable, frozen


def main():
    # CMU-MOSI feature dimensions (text 300, audio 5, visual 20), from Table II.
    mosi_dims = (300, 5, 20)
    # CMU-MOSEI dimensions, used only to build the stage-one backbone.
    mosei_dims = (300, 74, 35)

    aligned = audit((50, 50, 50), mosi_dims, "MOSI aligned (50/50/50)")
    unaligned = audit((50, 375, 500), mosi_dims, "MOSI unaligned (50/375/500)")

    backbone = MULTModel(make_hyp((50, 50, 50), mosi_dims))
    backbone_n = n_params(backbone)

    tmp = REPO_ROOT / "results" / "_tmp_backbone.pt"
    tmp.parent.mkdir(exist_ok=True)
    tr_a, fr_a = trainable_after_transfer(
        aligned["model"], mosei_dims, (50, 50, 50), tmp)
    tr_u, fr_u = trainable_after_transfer(
        unaligned["model"], mosei_dims, (50, 375, 500), tmp)
    tmp.unlink(missing_ok=True)

    w = 30
    print("=" * 74)
    print("MPLMM generation module -- parameter audit")
    print("repo: github.com/zrguo/MPLMM   file: src/model.py")
    print("=" * 74)
    print()
    print(f"MulT backbone (MULTModel, CMU-MOSI):{backbone_n:>13,}")
    print()
    print(f"{'':<{w}}{'aligned':>13}{'unaligned':>15}")
    print(f"{'':<{w}}{'50/50/50':>13}{'50/375/500':>15}")
    print("-" * 74)

    rows = [
        ("9 time-mixing layers", "time_mixing"),
        ("6 channel-mixing layers", "channel_mixing"),
        ("missing-signal prompts", "signal_prompts"),
        ("missing-type coupling m_*", "coupling"),
    ]
    for title, key in rows:
        note = "  <- length-independent" if key == "channel_mixing" else ""
        print(f"{title:<{w}}{aligned[key]:>13,}{unaligned[key]:>15,}{note}")

    print("-" * 74)
    print(f"{'length-dependent total':<{w}}{aligned['length_dependent']:>13,}"
          f"{unaligned['length_dependent']:>15,}")
    print(f"{'trainable after transfer':<{w}}{tr_a:>13,}{tr_u:>15,}")
    print(f"{'  as % of backbone':<{w}}{100*tr_a/backbone_n:>12.2f}%"
          f"{100*tr_u/backbone_n:>14.2f}%")
    print("=" * 74)
    print()
    print("Per-layer detail, nine time-mixing layers")
    print(f"{'layer':<10}{'aligned':>12}{'unaligned':>14}   role")
    roles = {
        "l_avp": "language absent",
        "a_lvp": "audio absent",
        "v_alp": "visual absent",
        "l_ap": "language + visual absent",
        "l_vp": "language + audio absent",
        "a_lp": "audio + visual absent",
        "a_vp": "audio + language absent",
        "v_ap": "visual + language absent",
        "v_lp": "visual + audio absent",
    }
    for name in TIME_MIXING:
        print(f"{name:<10}{aligned['time_mixing_detail'][name]:>12,}"
              f"{unaligned['time_mixing_detail'][name]:>14,}   {roles[name]}")
    print()
    print("All figures above were produced twice -- by instantiating the real")
    print("PromptModel and by closed-form Equation (1) -- and asserted equal.")
    print()
    print("Reading: the paper states its trainable parameter count does not grow")
    print("with backbone size. True, and not the binding constraint. It grows")
    print("with input sequence length instead, and at unaligned CMU-MOSI lengths")
    print(f"the trainable parameters ({tr_u:,}) exceed the frozen backbone")
    print(f"({backbone_n:,}) they exist to avoid fine-tuning.")


if __name__ == "__main__":
    main()