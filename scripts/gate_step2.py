"""Step-2 gate: the perception pipeline against the ground-truth expert on the same dev seeds.

    python scripts/gate_step2.py --truth results/perception/eval_expert_truth_setdown_legacy_dev.json \
        --perception results/perception/eval_perception_setdown_legacy_dev.json \
        --out results/perception/gate_step2_legacy.json

Pass: perception success >= truth success - 5 points and >= 75 % absolute, with the same pour mode and seeds.
Also reports the drawer-open sub-goal and the drawer estimate audit, and splits the perception-only failures by
whether the first failing step had already been retried (a retry state the CNN was not trained on) or failed on
its first attempt.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.stats import wilson  # noqa: E402

MARGIN, FLOOR = 0.05, 0.75


def first_failure(ep):
    """(step text, attempts, retried before) of the first step that failed, or None."""
    retried = False
    for s in ep["steps"]:
        if not s["ok"]:
            return s["text"], s["attempts"], retried
        retried = retried or s["attempts"] > 1
    bad = [k for k, v in ep["subgoals"].items() if not v]
    return (f"(all steps ok; sub-goal undone later: {','.join(bad)})", 0, retried) if bad else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", required=True)
    ap.add_argument("--perception", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    T, P = (json.load(open(p)) for p in (a.truth, a.perception))
    ts, ps = T["summary"], P["summary"]
    tep = {e["seed"]: e for e in T["episodes"]}
    pep = {e["seed"]: e for e in P["episodes"]}
    assert set(tep) == set(pep), "different seeds"
    assert ts.get("pour_mode") == ps.get("pour_mode"), (ts.get("pour_mode"), ps.get("pour_mode"))
    assert ps.get("perception") and not ts.get("perception")
    n = len(pep)
    kt, kp = sum(e["success"] for e in tep.values()), sum(e["success"] for e in pep.values())
    rt, rp = kt / n, kp / n
    both = sum(tep[s]["success"] and pep[s]["success"] for s in pep)
    only_t = sorted(s for s in pep if tep[s]["success"] and not pep[s]["success"])
    only_p = sorted(s for s in pep if pep[s]["success"] and not tep[s]["success"])
    lost = {}
    for s in only_t:
        txt, att, retried = first_failure(pep[s])
        lost[s] = dict(first_failure=txt, attempts=att, after_retry=retried,
                       observations=[(o["obj"], o["err_mm"]) for o in pep[s]["perception"]["observations"]])
    drawer = [d for e in pep.values() for d in e["perception"].get("drawer", [])]
    derr = sorted(abs(d["believed_mm"] - d["true_mm"]) for d in drawer)
    gate = dict(
        passed=bool(rp >= rt - MARGIN and rp >= FLOOR),
        rule=f"perception >= truth - {MARGIN * 100:.0f} points and >= {FLOOR * 100:.0f}% on the same seeds",
        pour_mode=ps["pour_mode"], n=n,
        truth=dict(file=a.truth, k=kt, rate=rt, ci95=list(wilson(kt, n)), code=ts.get("code")),
        perception=dict(file=a.perception, k=kp, rate=rp, ci95=list(wilson(kp, n)), code=ps.get("code"),
                        model=ps.get("perception"), drawer=ps.get("drawer")),
        gap_points=round((rt - rp) * 100, 1), both_succeed=both,
        truth_only=only_t, perception_only=only_p,
        subgoal_rates=dict(truth=ts["subgoal_rates"], perception=ps["subgoal_rates"]),
        drawer_audit=dict(n_releases=len(drawer),
                          abs_err_mm_median=derr[len(derr) // 2] if derr else None,
                          abs_err_mm_max=derr[-1] if derr else None),
        lost_by_perception=lost,
        # a failing step always shows 3 attempts; it counts as a retry state only if an earlier step was retried
        lost_after_retry=sum(v["after_retry"] for v in lost.values()),
    )
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(gate, f, indent=1)
    print(f"pour mode {gate['pour_mode']}: truth {kt}/{n}, perception {kp}/{n}, gap {gate['gap_points']} points "
          f"-> {'PASS' if gate['passed'] else 'FAIL'} ({gate['rule']})")
    print(f"drawer_open: truth {ts['subgoal_rates'].get('drawer_open')}, perception "
          f"{ps['subgoal_rates'].get('drawer_open')}; drawer estimate |err| median {gate['drawer_audit']['abs_err_mm_median']} mm")
    print(f"lost by perception: {len(only_t)} ({gate['lost_after_retry']} after a retry); gained: {len(only_p)}")
    for s, v in lost.items():
        print(f"  {s}: {v['first_failure']} (attempts {v['attempts']}, after retry {v['after_retry']})")


if __name__ == "__main__":
    main()
