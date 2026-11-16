# Changelog

Numbers come from the files named in each entry. Percent ranges are Wilson 95 % intervals.

## Phase 0 — baseline, tests, reproducibility (2026-11-15)

Reference numbers for everything that follows: `results/baseline/` (`python tasks.py baseline`), produced at
commit `f21d8ca` plus behaviour-neutral changes. Full tables: `results/baseline/REPORT.md`.

| Measurement | Stated before | Reproduced | Artifact |
|---|---|---|---|
| Scripted expert, seeds 0–9 | 10/10 (committed JSON: 9/10) | 10/10 (72–100 %) | `eval_expert_seeds0-9.json` |
| Scripted expert, seeds 10–19 | 7/10 | 9/10 (60–98 %) | `eval_expert_seeds10-19.json` |
| Scripted expert, seeds 1000–1029 | 27/30, no artifact | 28/30 (79–98 %) | `eval_expert_seeds1000-1029.json` |
| Learned plate+mug picks, INT8 iGPU, seeds 0–9 | 5/10 | 6, 5, 5 of 10 (16/30) | `eval_policy_int8_plate-mug_*`, `repeats/` |
| Learned all picks, INT8 iGPU, seeds 0–9 | 2/10, no artifact | 5, 5, 8 of 10 (18/30) | `eval_policy_int8_all_*`, `repeats/` |
| Same two settings on CPU | – | 6, 5 of 10 and 3, 4 of 10 | `repeats/*_cpu_r*.json` |
| PyTorch CPU latency | 6.97 ms | 8.45 ms | `benchmark.json` |
| OpenVINO CPU INT8 | 1.35 ms (5.2×) | 1.61 ms (5.2×) | `benchmark.json` |
| OpenVINO iGPU INT8 | 2.56 ms | 2.71 ms | `benchmark.json` |
| MuJoCo control step / 2-camera render | 1.5 / 1.9 ms | 1.93 / 1.97 ms | `benchmark.json` |
| INT8 max deviation from PyTorch | 0.043 (one random input) | median 0.16, worst 1.93 (12 real frames) | `ir_parity.json` |
| FP16 max deviation from PyTorch | 0.0004 (one call) | CPU: 0.0004 on the first call, up to 3.10 after; iGPU ≤ 0.015 | `ir_parity.json` |

Findings:
- Scripted-expert evaluation is deterministic (seeds 5 and 17 repeated bit for bit). The committed
  `results/eval_expert*.json` differ from the re-run because they predate the current code.
- Learned-policy evaluation is **not** repeatable, on the iGPU (outcome flipped on 4 of 10 seeds across three
  all-picks runs) or on the CPU (1 of 10). Cause not yet identified. The earlier conclusion that restricting the
  policy to plate+mug picks raises success from 20 % to 50 % is not supported by the repeats.
- The FP16 IR on the OpenVINO CPU device returns wrong outputs after its first inference. Not fixed yet.
- Training seeds are 100–399, so the dev split is 400–499 (`results/SPLITS.md`).
- Parallel iGPU evaluation could crash on the shared compile cache; fixed.
- Two parser defects pinned as expected failures: `hand the fork to arm A` gives an empty plan; an unknown
  object gives a pick step without an object.

Added: `tests/` (101 passing, 4 strict expected failures), `tasks.py`, `scripts/{stats,splits,env_info,report}.py`,
`deploy/parity.py`, `PLAN.md`, `results/SPLITS.md`. README numbers corrected to the reproduced values.

Not done in this phase: `docs/images/subgoal_success.png` and the slide deck still show the old evaluation numbers;
they are regenerated in Phase 8.

## Phase 1 — physics realism (in progress, 2026-11-16)

Artifacts in `results/phase1/`. Dev seeds 400–499 unless stated. Not finished: see the last list.

**Scene versions.** `scene="v2"` holds the object changes a friction pinch needs (rimmed plate, colliding mug
handle, cutlery 38 mm apart, 115 mm drawer); `grasp="contact"` only changes the gripper. Assist and contact are
compared on v2 only; the original scene is `legacy`.

**Per-object grasp trials, 100 dev seeds each** (`grasp_trials_dev.json`; pick, then carry with a shake, then
place; k of 100 with Wilson 95 % interval for the final stage). Contact-grasp tuning used seeds 400–431, so
these are tuned-on numbers; the test split is run once at the end of the phase.

| Object | contact@v2 pick / carried / placed | assist@v2 pick / carried / placed | assist@legacy placed (reference only) |
|---|---|---|---|
| Plate | 100 / 100 / 98 (93–99 %) | 100 / 100 / 100 (96–100 %) | 100 |
| Bottle | 98 / 93 / 86 (78–91 %) | 91 / 91 / 89 (81–94 %) | 87 |
| Mug | 99 / 99 / 99 (95–100 %) | 100 / 100 / 100 (96–100 %) | 100 |
| Spoon | 100 / 90 / 89 (81–94 %) | 77 / 77 / 76 (67–83 %) | 96 |
| Fork | 62 / 14 / 8 (4–15 %) | 100 / 100 / 100 (96–100 %) | 97 |

- Fork friction grasps are a documented limitation: 62 picked, 14 survived the carry. Not tuned further.
- The bottle is held, but it slides in the pinch until the pads reach its cap (seed 400: it follows 26 mm of a
  70 mm lift). Pinching the cap directly picked only 4/32, so the body grasp stays; pinned as an expected failure
  in `tests/test_contact.py`. The trial script now records lift shortfall; this file predates that field.
- Contact mode squeezes at the stock servo stall limit, 3.35 N·m. On seeds 400–431 the bottle was placed 1/32 at
  0.6, 16/32 at 1.2, 25/32 at 2.5 and 30/32 at 3.35 (scratch sweeps, not committed). A real STS3215 cannot hold
  stall torque for long.
- Assist on v2 is weaker than on legacy for the spoon (76 vs 96): the assist skills were given v2 values for the
  drawer pull and plate grasp only.

**Scripted expert, full task, weld assist, legacy scene:** 77/100 on dev (68–84 %), `eval_expert_assist_dev.json`.
The `wall_s` values in that file include time the laptop spent asleep.

**Learned-policy nondeterminism: source found** (`determinism_*.json`, `scripts/policy_determinism.py`).
Inference and physics are bit-exact. With identical simulator state the front camera differs by 1 intensity level
on at most 14 of 12,288 pixels, at multisampled shadow edges: 13 of 18 run pairs diverged with default rendering,
0 of 28 with shadows off, 0 of 28 with multisampling off. `--no-msaa` renders reproducibly but is not what the
current policy was trained on; its effect on success is not measured yet.

**Learned policy on dev, INT8 on iGPU, plate+mug picks:** one run of 100 so far, 34/100 (25–44 %),
`policy_dev/eval_policy_int8_plate-mug_dev_r1.json`. This is one run, not the required mean of three; the other
runs were stopped when the machine ran out of memory.

**Evaluation robustness.** Per-episode wall-clock budget (suspended time not charged) and simulated-step cap;
episodes that raise are recorded and counted instead of aborting the run.

Not done yet in this phase: learned-policy runs 2–3 and the two `--no-msaa` runs; full-task expert with contact
grasps on v2; actuator and sensor realism; the randomization config file; the particle pour check; the test-split
grasp trials.
