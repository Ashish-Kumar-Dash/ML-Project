"""Dead missing-type prompt: released zero-init vs proposed N(0, 0.02) repair.

Released code (third_party/MPLMM/src/model.py:225-230) zero-initialises
missing_type_prompt, m_a, m_v, m_l. batch_prompt = missing_type_prompt @ mp with
mp built from prompt*_{m,nm} @ m_* (model.py:340-380), so both factors of the
bilinear product start at zero and their gradients stay exactly zero.
The repair re-initialises only those four tensors to N(0, 0.02).
Exits non-zero if either arm does not behave as stated.
"""
import json, sys, torch
sys.path.insert(0, "."); import lenmod.compat  # noqa: E402,F401
from lenmod.config import make_hp            # noqa: E402
from src.model import PromptModel            # noqa: E402

NAMES = ("missing_type_prompt", "m_a", "m_v", "m_l")

def run(repair, steps=50, seed=1):
    torch.manual_seed(seed)
    model = PromptModel(make_hp(50, 50, 50)); model.train()
    if repair:
        with torch.no_grad():
            for n in NAMES: getattr(model, n).normal_(0.0, 0.02)
    init = {n: getattr(model, n).detach().clone() for n in NAMES}
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    g = torch.Generator().manual_seed(seed)
    B = 14
    x_l, x_a, x_v = (torch.randn(B, 50, d, generator=g) for d in (300, 5, 20))
    y = torch.randn(B, 1, generator=g)
    mods = [i % 7 for i in range(B)]
    first_grad = None; first_nz = None
    for s in range(steps):
        opt.zero_grad()
        loss = torch.nn.functional.l1_loss(model(x_l, x_a, x_v, mods), y)
        loss.backward()
        if first_grad is None:
            first_grad = {n: float(getattr(model, n).grad.abs().max()) for n in NAMES}
        if first_nz is None and any(float(getattr(model, n).grad.abs().max()) > 0 for n in NAMES):
            first_nz = s + 1
        opt.step()
    model.eval()
    with torch.no_grad():
        out = model(x_l, x_a, x_v, mods)
        saved = model.missing_type_prompt.clone()
        model.missing_type_prompt.zero_()
        out_ablate = model(x_l, x_a, x_v, mods)
        model.missing_type_prompt.copy_(saved)
    return {
        "first_step_max_abs_grad": first_grad,
        "final_max_abs": {n: float(getattr(model, n).detach().abs().max()) for n in NAMES},
        "max_abs_change_from_init": {n: float((getattr(model, n).detach() - init[n]).abs().max()) for n in NAMES},
        "first_step_with_nonzero_grad": first_nz,
        "mp_max_abs": float(model.mp.abs().max()),
        "output_change_if_missing_type_prompt_zeroed": float((out - out_ablate).abs().max()),
    }

res = {"released_zero_init": run(False), "repair_normal_0p02": run(True)}
print(json.dumps(res, indent=1))
rel, rep = res["released_zero_init"], res["repair_normal_0p02"]
assert all(v == 0.0 for v in rel["final_max_abs"].values()), "released arm moved"
assert rel["mp_max_abs"] == 0.0 and rel["output_change_if_missing_type_prompt_zeroed"] == 0.0
assert all(v > 0.0 for v in rep["max_abs_change_from_init"].values()), "repair: a tensor did not train"
assert all(v == 0.0 for v in rel["max_abs_change_from_init"].values())
assert rep["mp_max_abs"] > 0.0, "repair: mp still zero"
assert rep["output_change_if_missing_type_prompt_zeroed"] > 0.0, "repair: prompt has no effect"
print("OK: released prompt is inert; repaired prompt trains and changes the output")
