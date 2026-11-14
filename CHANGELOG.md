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
