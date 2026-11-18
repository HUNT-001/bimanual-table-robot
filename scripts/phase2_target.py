"""The Phase 2 target and the comparison of a candidate policy against it, per object.

Target: first-attempt pick success of the learned policy on the v2 scene, hybrid grasp (contact, fork on weld
assist), realism `nominal`, dev seeds, `--no-msaa`, must be at least the level the existing policy reaches on the
legacy scene with weld assist and no realism (the same dev seeds, `--no-msaa`), for each object separately.

    python scripts/phase2_target.py --write results/phase2/target.json
    python scripts/phase2_target.py --candidate results/phase2/picks_B.json [--candidate ...]

"Met" means the candidate's point estimate is at least the target. Because 100 seeds cannot separate small
differences, the table also says whether the candidate is below the target beyond sampling noise (its upper 95 %
bound is under the target), and the Wilson interval is always shown.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.stats import wilson  # noqa: E402

LEGACY_CLEAN_RUNS = [f"results/phase1/policy_dev/eval_policy_int8_plate-mug_dev_nomsaa_r{i}.json" for i in (1, 2)]
OBJECTS = ("plate", "mug")


def first_try(path):
    """{object: (k, n)} first-attempt successes of the learned steps in one evaluation file."""
    out = {}
    for ep in json.load(open(path))["episodes"]:
        for st in ep["steps"]:
            if st["executor"] == "policy":
                obj = st["text"].split()[-1]
                k, n = out.get(obj, (0, 0))
                out[obj] = (k + int(st["ok"] and st["attempts"] == 1), n + 1)
    return out


def legacy_clean_target(root="."):
    pooled = {}
    for r in LEGACY_CLEAN_RUNS:
        s = json.load(open(os.path.join(root, r)))["summary"]
        assert (s["scene"], s["grasp"], s["msaa"]) == ("legacy", "assist", False), r
        for o, (k, n) in first_try(os.path.join(root, r)).items():
            a, b = pooled.get(o, (0, 0))
            pooled[o] = (a + k, b + n)
    return {o: dict(k=k, n=n, rate=k / n, ci95=list(wilson(k, n))) for o, (k, n) in sorted(pooled.items())}


def compare(candidate_path, target):
    s = json.load(open(candidate_path))["summary"]
    rows = []
    for o, (k, n) in sorted(first_try(candidate_path).items()):
        lo, hi = wilson(k, n)
        t = target[o]["rate"]
        rows.append(dict(object=o, k=k, n=n, rate=k / n, ci95=[lo, hi], target=t,
                         met=k / n >= t, below_beyond_noise=hi < t))
    return s, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", default=None, help="write the target (with its source runs) to this JSON file")
    ap.add_argument("--candidate", action="append", default=[], help="evaluation JSON from evaluate.py")
    a = ap.parse_args()
    target = legacy_clean_target()
    print("target (legacy scene, weld assist, no realism, --no-msaa; pooled over runs r1, r2):")
    for o, v in target.items():
        print(f"  {o:6s} {v['k']:3d}/{v['n']} = {100 * v['rate']:.1f} %  (95 % CI {100 * v['ci95'][0]:.1f}-{100 * v['ci95'][1]:.1f})")
    if a.write:
        os.makedirs(os.path.dirname(os.path.abspath(a.write)), exist_ok=True)
        json.dump(dict(definition=__doc__.strip().split("\n\n")[0], runs=LEGACY_CLEAN_RUNS, target=target),
                  open(a.write, "w"), indent=1)
    for c in a.candidate:
        s, rows = compare(c, target)
        print(f"\n{c}\n  {s['scene']} / {s['grasp']}"
              f"{'+assist(' + ','.join(s['assist_objects']) + ')' if s.get('assist_objects') else ''} / "
              f"realism={s.get('realism')} / msaa={s['msaa']} / code {s.get('code', {}).get('commit')}")
        for r in rows:
            flag = "MET" if r["met"] else ("below target beyond noise" if r["below_beyond_noise"] else "below target, within noise")
            print(f"  {r['object']:6s} {r['k']:3d}/{r['n']} = {100 * r['rate']:5.1f} % (95 % CI {100 * r['ci95'][0]:.1f}-{100 * r['ci95'][1]:.1f})"
                  f"  target {100 * r['target']:.1f} %  -> {flag}")


if __name__ == "__main__":
    main()
