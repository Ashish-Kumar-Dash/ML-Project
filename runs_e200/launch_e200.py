#!/usr/bin/env python3
"""200-epoch ladder launcher (Tanmay, 2026-10-27).

Runs the team's run.py UNCHANGED except for one override: hyp_params.when
(ReduceLROnPlateau patience) = 30 instead of the hard-coded 10 in
lenmod/config.py. Epochs are passed on the command line (--num_epochs 200).
After run.py finishes, the results.json is annotated with the override so the
file is self-describing.
"""
import json, os, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
os.chdir(REPO); sys.path.insert(0, str(REPO))
WHEN = 30
import run  # noqa: E402  (team's run.py, not modified)
_orig_make_hp = run.make_hp
def _make_hp_when(*a, **k):
    hp = _orig_make_hp(*a, **k)
    hp.when = WHEN
    print(f"[LAUNCHER] hp.when overridden 10 -> {hp.when}; num_epochs={hp.num_epochs}", flush=True)
    return hp
run.make_hp = _make_hp_when
sys.argv = [str(REPO / "run.py")] + sys.argv[1:]
run.main()
name = sys.argv[sys.argv.index("--name") + 1]
rd = Path(name) if Path(name).is_absolute() else REPO / name
jp = rd / "results.json"
d = json.loads(jp.read_text())
d.setdefault("hyperparameters", {})["when"] = WHEN
d["launcher"] = {"script": "runs_e200/launch_e200.py", "overrides": {"when": WHEN},
                 "note": "run.py unchanged; make_hp wrapped to set when=30"}
jp.write_text(json.dumps(d, indent=2))
print(f"[LAUNCHER] annotated {jp}", flush=True)
