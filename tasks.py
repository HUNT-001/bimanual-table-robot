"""Cross-platform task runner (Windows and Linux; no make required).

    python tasks.py test [--fast]            unit tests (--fast skips full-episode tests)
    python tasks.py eval --split dev         scripted expert on a seed split -> results/eval_expert_dev.json
    python tasks.py eval --split dev --policy models/act_pick_int8.xml --device GPU
    python tasks.py bench                    latency benchmark + IR parity -> results/
    python tasks.py report [--dir results]   Markdown tables with confidence intervals
    python tasks.py baseline                 re-create results/baseline/ (the Phase 0 reference numbers)
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from scripts.splits import SPLITS  # noqa: E402

PY = sys.executable
IRS = [f"models/act_pick_{p}.xml" for p in ("fp32", "fp16", "int8")]


def run(*cmd, log=None):
    print("$", " ".join(cmd), flush=True)
    if log:
        os.makedirs(os.path.dirname(os.path.join(ROOT, log)), exist_ok=True)
        with open(os.path.join(ROOT, log), "w") as f:
            rc = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
    else:
        rc = subprocess.run(cmd, cwd=ROOT).returncode
    if rc:
        sys.exit(rc)


def evaluate(seeds, out, workers, policy=None, device="GPU", policy_objects="plate,mug", log=None):
    cmd = [PY, "scripts/evaluate.py", "--seeds", seeds, "--workers", str(workers), "--out", out]
    if policy:
        cmd += ["--policy", policy, "--device", device, "--policy-objects", policy_objects]
    run(*cmd, log=log)


def bench(out_dir):
    run(PY, "-m", "deploy.benchmark", "--models", *IRS, "--torch", "models/act_pick.pt", "--sim",
        "--out", f"{out_dir}/benchmark.json")
    run(PY, "-m", "deploy.parity", "--models", *IRS, "--out", f"{out_dir}/ir_parity.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="task", required=True)
    t = sub.add_parser("test")
    t.add_argument("--fast", action="store_true")
    e = sub.add_parser("eval")
    e.add_argument("--split", choices=sorted(SPLITS), default="dev")
    e.add_argument("--policy", default=None)
    e.add_argument("--device", default="GPU")
    e.add_argument("--policy-objects", default="plate,mug")
    e.add_argument("--workers", type=int, default=8)
    e.add_argument("--out", default=None)
    b = sub.add_parser("bench")
    b.add_argument("--out-dir", default="results")
    r = sub.add_parser("report")
    r.add_argument("--dir", default="results")
    bl = sub.add_parser("baseline")
    bl.add_argument("--workers", type=int, default=8)
    bl.add_argument("--skip-evals", action="store_true")
    a = ap.parse_args()

    if a.task == "test":
        run(PY, "-m", "pytest", "tests", "-q", *(["-m", "not slow"] if a.fast else []))
    elif a.task == "eval":
        if a.split in ("train", "perception"):
            sys.exit("refusing to evaluate on training seeds (see results/SPLITS.md)")
        name = "expert" if not a.policy else os.path.splitext(os.path.basename(a.policy))[0]
        out = a.out or f"results/eval_{name}_{a.split}.json"
        evaluate(SPLITS[a.split], out, a.workers, a.policy, a.device, a.policy_objects)
    elif a.task == "bench":
        bench(a.out_dir)
    elif a.task == "report":
        run(PY, "scripts/report.py", "--dir", a.dir)
    elif a.task == "baseline":
        B, w = "results/baseline", a.workers
        if not a.skip_evals:
            for seeds in ("0-9", "10-19", "1000-1029"):
                evaluate(seeds, f"{B}/eval_expert_seeds{seeds}.json", w, log=f"{B}/logs/eval_expert_seeds{seeds}.log")
            for objs, tag in (("plate,mug", "plate-mug"), ("all", "all")):
                evaluate("0-9", f"{B}/eval_policy_int8_{tag}_seeds0-9.json", w, IRS[2], "GPU", objs,
                         log=f"{B}/logs/eval_policy_int8_{tag}_seeds0-9.log")
            # learned-policy runs are not repeatable run to run; repeat them to size the spread
            for objs, tag in (("plate,mug", "plate-mug"), ("all", "all")):
                for rep, dev in (("gpu_r2", "GPU"), ("gpu_r3", "GPU"), ("cpu_r1", "CPU"), ("cpu_r2", "CPU")):
                    evaluate("0-9", f"{B}/repeats/eval_policy_int8_{tag}_seeds0-9_{rep}.json", w, IRS[2], dev, objs,
                             log=f"{B}/repeats/logs/{tag}_{rep}.log")
        bench(B)          # run with nothing else active on the machine
        run(PY, "scripts/env_info.py", "--out", f"{B}/environment.json")
        run(PY, "scripts/report.py", "--dir", B)


if __name__ == "__main__":
    main()
