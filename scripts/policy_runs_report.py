"""Summarise repeated evaluations of the same configuration on the same seeds.

Learned-policy runs are not repeatable run to run, so a result is the mean over runs. The interval is a cluster
bootstrap over seeds (a seed's outcomes in different runs are correlated, so episodes are not independent draws).

    python scripts/policy_runs_report.py --name "INT8 iGPU, learned plate+mug picks" \
        --runs results/phase1/policy_dev/eval_policy_int8_plate-mug_dev_r*.json --out results/phase1/policy_dev/summary_default
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.stats import wilson  # noqa: E402


def cluster_bootstrap(m, n_boot=5000, seed=0):
    """m: runs x seeds matrix of 0/1. 95 % percentile interval of the grand mean, resampling seeds."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, m.shape[1], (n_boot, m.shape[1]))
    means = np.sort(m[:, idx].mean(axis=(0, 2)))
    return float(means[int(0.025 * n_boot)]), float(means[int(0.975 * n_boot) - 1])


def summarise(files, name):
    runs = [json.load(open(f)) for f in files]
    seeds = [e["seed"] for e in runs[0]["episodes"]]
    assert all([e["seed"] for e in r["episodes"]] == seeds for r in runs), "runs must cover the same seeds"
    s0 = runs[0]["summary"]
    for key in ("policy", "device", "policy_objects", "grasp", "scene", "msaa"):
        assert all(r["summary"].get(key) == s0.get(key) for r in runs), f"runs differ in {key}"
    m = np.array([[e["success"] for e in r["episodes"]] for r in runs], float)
    goals = sorted(runs[0]["summary"]["subgoal_rates"])
    out = dict(name=name, files=[os.path.basename(f) for f in files], n_runs=len(runs), n_seeds=len(seeds),
               seeds=f"{seeds[0]}-{seeds[-1]}", config={k: s0.get(k) for k in ("policy", "device", "policy_objects",
                                                                                "grasp", "scene", "msaa")},
               success_per_run=[int(x) for x in m.sum(1)],
               mean_success=float(m.mean()), std_across_runs=float(m.mean(1).std(ddof=1)) if len(runs) > 1 else 0.0,
               ci95_cluster_bootstrap=list(cluster_bootstrap(m)),
               per_run_wilson=[list(wilson(int(x), len(seeds))) for x in m.sum(1)],
               seeds_always_success=int((m.min(0) == 1).sum()), seeds_never_success=int((m.max(0) == 0).sum()),
               seeds_mixed=int(((m.min(0) == 0) & (m.max(0) == 1)).sum()),
               identical_episodes=int(sum(len({json.dumps(r["episodes"][i]["steps"]) + str(r["episodes"][i]["sim_time_s"])
                                               for r in runs}) == 1 for i in range(len(seeds)))),
               timed_out=[r["summary"].get("n_timed_out", 0) for r in runs],
               errors=[r["summary"].get("n_errors", 0) for r in runs],
               subgoal_mean={g: float(np.mean([r["summary"]["subgoal_rates"][g] for r in runs])) for g in goals})
    pol = [r["summary"] for r in runs if "policy_steps" in r["summary"]]
    if pol:
        out["policy_first_try_success_mean"] = float(np.mean([p["policy_first_try_success"] for p in pol]))
        out["policy_steps_per_run"] = [p["policy_steps"] for p in pol]
        out["policy_latency_ms_mean"] = float(np.mean([p["policy_latency_ms_mean"] for p in pol]))
    return out


def markdown(o):
    lo, hi = o["ci95_cluster_bootstrap"]
    n = o["n_seeds"]
    md = [f"# {o['name']}", "",
          f"{o['n_runs']} runs x {n} seeds ({o['seeds']}); config: {o['config']}", "",
          f"**Full-task success: {100 * o['mean_success']:.1f} %** (95 % CI {100 * lo:.1f}-{100 * hi:.1f}, cluster "
          f"bootstrap over seeds); std across runs {100 * o['std_across_runs']:.1f} points", "",
          "| Run | Successes | Wilson 95 % | Timed out | Errors |", "|---|---|---|---|---|"]
    for i, k in enumerate(o["success_per_run"]):
        w = o["per_run_wilson"][i]
        md.append(f"| {o['files'][i]} | {k}/{n} | {100 * w[0]:.0f}-{100 * w[1]:.0f} % | {o['timed_out'][i]} | {o['errors'][i]} |")
    md += ["", f"Per seed across runs: always succeeded {o['seeds_always_success']}, never {o['seeds_never_success']}, "
               f"mixed {o['seeds_mixed']}; episodes identical step for step in every run: {o['identical_episodes']}/{n}.",
           "", "Sub-goal rates (mean over runs): " + ", ".join(f"{g} {100 * v:.0f} %" for g, v in o["subgoal_mean"].items())]
    if "policy_first_try_success_mean" in o:
        md += ["", f"Learned steps solved on the first attempt, no scripted fallback: "
                   f"{100 * o['policy_first_try_success_mean']:.0f} % (mean over runs; {o['policy_steps_per_run']} steps); "
                   f"mean inference {o['policy_latency_ms_mean']:.1f} ms."]
    return "\n".join(md) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True, help="path stem; writes <stem>.json and <stem>.md")
    a = ap.parse_args()
    files = sorted(f for pat in a.runs for f in glob.glob(pat))
    o = summarise(files, a.name)
    json.dump(o, open(a.out + ".json", "w"), indent=1)
    open(a.out + ".md", "w", encoding="utf-8").write(markdown(o))
    print(markdown(o))


if __name__ == "__main__":
    main()
