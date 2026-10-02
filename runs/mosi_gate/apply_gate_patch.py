import re, sys
p = "src/train.py"; s = open(p).read()
dp = "net = nn.DataParallel(model) if batch_size > 10 else model"
assert s.count(dp) == 2, s.count(dp)
s = s.replace(dp, "net = model  # single GPU: DataParallel removed")
old = "    _, results, truths = evaluate(model, criterion, test=False)\n"
assert s.count(old) == 1
block = r"""    # ---- GATE: evaluate best checkpoint on valid AND test from the same RNG state ----
    import random as _r, numpy as _np, json as _json, os as _os
    _st = (_r.getstate(), _np.random.get_state(), torch.get_rng_state(),
           torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    def _restore():
        _r.setstate(_st[0]); _np.random.set_state(_st[1]); torch.set_rng_state(_st[2])
        if _st[3] is not None: torch.cuda.set_rng_state_all(_st[3])
    _out = _os.environ.get("GATE_OUT", ".")
    for _split, _flag in (("valid", False), ("test", True)):
        _restore()
        _l, _res, _tru = evaluate(model, criterion, test=_flag)
        print(f"==== GATE final eval split={_split} test={_flag} loss={float(_l):.4f} n={_res.numel()}", flush=True)
        eval_mosi(_res, _tru, True)
        _np.savez(_os.path.join(_out, f"preds_{_split}.npz"),
                  preds=_res.view(-1).cpu().numpy(), truths=_tru.view(-1).cpu().numpy())
    _rows = []
    for _k in range(int(_os.environ.get("GATE_DRAWS", "20"))):
        for _split, _flag in (("valid", False), ("test", True)):
            _r.seed(1000 + _k)
            _l, _res, _tru = evaluate(model, criterion, test=_flag)
            _p = _res.view(-1).cpu().numpy(); _t = _tru.view(-1).cpu().numpy(); _nz = _t != 0
            _rows.append(dict(draw=_k, split=_split, n=int(_p.size), mae=float(_np.abs(_p - _t).mean()),
                              acc2_nonzero=float(((_p[_nz] > 0) == (_t[_nz] > 0)).mean()),
                              acc2_has0=float(((_p >= 0) == (_t >= 0)).mean())))
    _json.dump(_rows, open(_os.path.join(_out, "mask_draws.json"), "w"), indent=1)
    _restore()
    print("==== UPSTREAM final eval (line 160, test=False) follows", flush=True)
    # ---- end GATE ----
"""
s = s.replace(old, block + old)
open(p, "w").write(s)
print("patched")
