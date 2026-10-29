#!/usr/bin/env python3
"""Byte-identity check: native seed-1 rerun (runs_e200_native_v2, HEAD) vs original (runs_e200, ebbcb4e), same machine.
Compares every per-epoch training record (except wall-clock duration), best epoch/val loss, trainable
parameter count, and all test results. Identity => the commit change did not touch the A/B training path."""
import json, sys
from pathlib import Path
R = Path("/home/user/Downloads/ML-project/ML-Project")
out = []
ok_all = True
for arm in ("baseline", "invariant"):
    old = json.loads((R / f"runs_e200/mosi_ladder_native_{arm}_seed1/results.json").read_text())
    new = json.loads((R / f"runs_e200_native_v2/mosi_ladder_native_{arm}_seed1/results.json").read_text())
    strip = lambda h: [{k: v for k, v in e.items() if k != "duration"} for e in h]
    ho, hn = strip(old["training_history"]), strip(new["training_history"])
    first = next((i + 1 for i, (a, b) in enumerate(zip(ho, hn)) if a != b), None)
    checks = {"history_len": len(ho) == len(hn), "history_identical": first is None and len(ho) == len(hn),
              "best_epoch": old["training_outcome"]["best_epoch"] == new["training_outcome"]["best_epoch"],
              "best_val_loss": old["training_outcome"]["best_val_loss"] == new["training_outcome"]["best_val_loss"],
              "trainable_parameters": old["pipeline_checks"]["trainable_parameters"] == new["pipeline_checks"]["trainable_parameters"],
              "test_results": old["test_results"] == new["test_results"]}
    ok = all(checks.values()); ok_all &= ok
    out.append(f"{arm}: {'IDENTICAL' if ok else 'DIFFERS'} | old commit {old['project_commit'][:7]} new {new['project_commit'][:7]} | "
               f"first differing epoch: {first} | six-case old {old['test_results']['mean_cases_0_to_5']['acc2']:.6f} new {new['test_results']['mean_cases_0_to_5']['acc2']:.6f} | {checks}")
out.append("OVERALL: " + ("PASS - byte-identical on both arms" if ok_all else "FAIL - see above"))
(R / "results/native_v2_seed1_identity.txt").write_text("\n".join(out) + "\n"); print("\n".join(out))
