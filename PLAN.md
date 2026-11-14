# PLAN — bimanual-table: optimize, harden, extend, present

Written 2026-11-15 after reading the repo at `f21d8ca`. Estimates are estimates; every measured number will live
under `results/` and be produced by a documented command.

## 1. What I verified, and where the brief and the repo disagree

| Brief says | Repo shows | What I will do |
|---|---|---|
| Training seeds 0–299, dev 300–399 | Demos are `data/demos/ep_00100…ep_00399` (300 files); README and `run_all.sh` say 100–399 | **Splits: train 100–399, dev 400–499, test 1000–1099.** Dev 300–399 would be training data. Seeds 0–99 are "legacy development" seeds (scripted skills were tuned on 0–9) and are used only to reproduce old tables. |
| Scripted expert 10/10 on seeds 0–9 | Committed `results/eval_expert.json` is **9/10** (seed 5: drawer) | Re-run; report whatever reproduces; correct the README if it stays 9/10. |
| 27/30 on 1000–1029 | Row exists only in an uncommitted README edit; **no artifact** | Re-run into `results/baseline/`. Seeds 1000–1029 have already been looked at for the scripted expert, so the final test report will show 1000–1029 and 1030–1099 separately as well as pooled. |
| Learned all-picks 2/10 | No artifact in `results/` | Re-run. |
| `LLMParser` is a stub | Confirmed; `openvino_genai` is not installed | Stays "stub" in the README until Phase 4 evaluates it. |
| Pour: tilt > 55° | Code uses 55°; the docstring says 70° | Fix the docstring. |

Other facts that shape the plan:
- Val split in `policy/train.py` is the first 10 % of files (seeds 100–129), unaugmented. No leakage found.
- `DemoDataset` loads every shard into RAM (3.4 GB compressed on disk, far more decompressed). Phase 2 replaces this.
- An expert episode costs about 60 s wall per worker; 100 seeds at 8 workers is about 12–15 min.
- Environment: Python 3.13.5, torch 2.6.0+cu124, MuJoCo 3.13.0, OpenVINO 2026.3.1, NNCF 3.3.0, pytest 9.0.1.
  Devices: CPU, GPU.0 (Intel iGPU), GPU.1 (RTX 4060). `lerobot`, `openvino_genai`, `zarr` are not installed.
- The weld assist also drives the drawer pull and the hand-over transfer, so "contact grasps" touches four skills,
  not just pick.

## 2. Phases

GPU hours are RTX 4060 training time; CPU hours are simulation at 8 workers.

| Phase | Deliverables | Compute (est.) | Exit criteria |
|---|---|---|---|
| **0 Baseline, tests** | `results/baseline/*` (evals, benchmark, env info), `results/SPLITS.md`, `tests/`, `tasks.py` (`test`, `eval`, `bench`, `report`) | 1 h CPU | Baselines reproduce within noise or the difference is explained; tests pass |
| **1 Physics** | `--grasp contact\|assist`; pad geometry and friction tuning; actuator/sensor realism; `sim/randomization.yaml`; particle pour metric | 3–5 h CPU | Per-object pick success, ≥ 100 trials each, Wilson CIs, both modes; expert on test seeds with contact grasps |
| **2 Data engine** | Streaming shard dataset with manifest; recovery data; DAgger ≥ 3 iterations; `results/failure_taxonomy.md` | 8–10 h GPU, 3 h CPU | Success vs data size and vs DAgger iteration, with CIs |
| **3 Models** | ACT-Lite variants (wrist cam, state history, size, chunk, ensembling), flow-matching head, SmolVLA attempt, learned place/hand-off/drawer/pour | 14–20 h GPU, 4 h CPU | One comparison table; best picked on dev; test reported once |
| **4 Perception** | Vision verifier (`--verifier vision\|oracle`), vision object localization, 60+ command set, LLM planner evaluation | 2–3 h GPU, 2 h CPU | End-to-end success with vision vs oracle, gap quantified |
| **5 Optimization** | Per-stage latency p50/p95/p99 over ≥ 1,000 steps; optimization ledger; accuracy gate | 1–2 h CPU/iGPU | Ledger: change → latency delta → success delta → kept/rejected |
| **6 Robustness, safety** | Perturbation sweeps; guards, timeouts, e-stop, fail-safe; JSONL episode logs | 3–4 h CPU | Curves plus a list of unhandled cases |
| **7 Hardware readiness** | `robot/` interface with dry-run, teleop collection, calibration docs, BOM, staged test plan | negligible | Dry-run passes on the sim backend |
| **8 Feasibility, presentation** | `docs/FEASIBILITY.md`, `docs/slides.pdf`, `docs/one_pager.pdf`, README rewrite, claims table, demo script | 0.5 h CPU | Every slide rendered to PNG and inspected; every README number traced to an artifact |

Total: roughly 25–35 GPU hours and 15–20 CPU hours. This is several working sessions; long jobs are resumable and
logged under `results/<phase>/logs/`.

## 3. Method rules I will hold to

- Success rates: Wilson 95 % intervals. Paired comparisons on identical seeds also report the per-seed win/loss
  counts, since ten-seed intervals are too wide to separate most variants.
- Dev (400–499) for every choice; test (1000–1099) once, at the end of Phase 3 and for the final tables.
- Each phase ends with: `results/` updated, tests green, a `CHANGELOG.md` entry with before/after, one commit per
  concern.
- The current pipeline stays runnable at each commit; new behaviour is behind flags whose defaults reproduce today's.

## 4. Risks, most likely first

1. **Contact grasps may not reach assist-level success.** SO-101 jaws are thin meshes; the plate (rim pinch) and
   cutlery in the drawer are the hard cases. If a class stays poor, I report it with CIs, keep the assist as the
   fallback for that class, and say so. Everything downstream (data, models) is then reported for both modes only
   where compute allows; otherwise in the mode chosen at the end of Phase 1.
2. **Phase 1 invalidates the existing 300 demos.** Changing actuator and contact physics changes expert
   trajectories, so data is re-collected after Phase 1 (about 20 min) and the ACT-Lite baseline retrained (about
   65 min) so Phase 2/3 comparisons are like for like.
3. **SmolVLA on 8 GB VRAM, Windows, Python 3.13.** May not install or fit. Time-boxed to about 6 h of effort;
   reported as "tried, did not fit" with the evidence if so.
4. **Learned skills beyond pick** (pour and hand-off are long and bimanual). Gated one at a time; likely that not
   all four pass the "scripted − 5 points" gate.
5. **Vision localization at 96×128** is coarse; may need a higher-resolution perception render, which costs latency.
6. **Run-to-run variance** in latency on a laptop. Benchmarks are run with nothing else active, repeated, and
   reported with spread.

## 5. Decisions I need from you (none block Phase 0–1)

- **Downloads and installs (Phase 3–4):** `lerobot`, SmolVLA weights, `openvino-genai`, and a small quantized LLM
  (order of 0.5–1.5 GB each). No money, but disk and network. I will use a separate virtual environment so the
  current one stays intact, unless you object.
- **Splits as corrected above** (train 100–399, dev 400–499). Say so if you would rather re-collect on 0–299.
- **Scope:** if Phase 3's compute runs long, I will propose cutting model variants before cutting evaluation size.
