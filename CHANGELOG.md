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

## Phase 1 — physics realism (2026-11-16 to 2026-11-17)

Artifacts in `results/phase1/`. Dev seeds 400–499 unless stated.

**Scene versions.** `scene="v2"` holds the object changes a friction pinch needs (rimmed plate, colliding mug
handle, cutlery 38 mm apart, 115 mm drawer); `grasp="contact"` only changes the gripper. Assist and contact are
compared on v2 only; the original scene is `legacy`.

**Per-object grasp trials, 100 dev seeds each** (pick, then carry with a wrist shake, then place; k of 100 with
Wilson 95 % interval for the final stage). Contact rows: `grasp_trials_dev_contact_fixedshake.json` (commit
9c73538); assist rows: `grasp_trials_dev.json` (b25b90a; the shake fix does not change assist mode). Contact
tuning used seeds 400–431, so these are tuned-on numbers; the test split is run once at the end of the phase.

| Object | contact@v2 pick / carried / placed | assist@v2 pick / carried / placed | assist@legacy placed (reference only) |
|---|---|---|---|
| Plate | 100 / 100 / 100 (96–100 %) | 100 / 100 / 100 (96–100 %) | 100 |
| Bottle | 100 / 100 / 68 (58–76 %) | 91 / 91 / 89 (81–94 %) | 87 |
| Mug | 99 / 99 / 99 (95–100 %) | 100 / 100 / 100 (96–100 %) | 100 |
| Spoon | 100 / 0 / 0 (0–4 %) | 77 / 77 / 76 (67–83 %) | 96 |
| Fork | 55 / 0 / 0 (0–4 %) | 100 / 100 / 100 (96–100 %) | 97 |

- Cutlery friction grasps are a documented limitation and are not tuned further. Both are picked (spoon 100,
  fork 55) but none survives the shake. An earlier run (57fbc3d, before the half-speed carry) carried the spoon
  90/100. With the shake held identical, seed 400 keeps the spoon after a full-speed carry and loses it on the
  third shake after a half-speed one (6/6 vs 0/6 on seeds 400–405, scratch runs). The pinch is marginal: small
  differences in the state the carry leaves decide it. In the full task, which has no shake, contact places the
  spoon in 84/100 episodes.
- Bottle lift now follows the gripper (median 63.7 mm of a 70 mm lift, minimum 62.1) after the centred descent
  (43bdb1b); 32 of 100 fail at placing.
- Until 9c73538 the half-speed carry also slowed the trial's shake in contact mode only, so contact and assist
  were not shaken alike; `grasp_trials_dev.json` contact rows are from that version and are superseded.
- Contact mode squeezes at the stock servo stall limit, 3.35 N·m. On seeds 400–431 the bottle was placed 1/32 at
  0.6, 16/32 at 1.2, 25/32 at 2.5 and 30/32 at 3.35 (scratch sweeps, not committed). A real STS3215 cannot hold
  stall torque for long.
- Assist on v2 is weaker than on legacy for the spoon (76 vs 96): the assist skills were given v2 values for the
  drawer pull and plate grasp only.

**Scripted expert, full task, weld assist, legacy scene:** 77/100 on dev (68–84 %), `eval_expert_assist_dev.json`.
The `wall_s` values in that file include time the laptop spent asleep.

**Scripted expert, full task, v2 scene, dev seeds** (`fulltask_dev/*.json`, commit b25b90a; 0 timeouts, 0 errors
in every run). Failure causes are the first failed step of each failed episode.

| Grasp | Success | Main failure causes (episodes) |
|---|---|---|
| Weld assist | 65/100 (55–74 %) | bottle put-back 14, spoon pick 10, pour 10, bottle re-pick 1 |
| Contact, fork on weld assist | 65/100 (55–74 %) | plate place 13, pour 8, spoon pick 6, re-picks 4, mug place 2, other 2 |
| Contact (friction only) | 11/100 (6–19 %) | fork pick 44, fork hand-over 22, plate place 13, spoon pick 6, other 4 |

- Pure contact fails on the fork in 66 of its 89 failures, as the grasp trials predict. Putting only the fork on
  the weld assist brings contact level with full assist (65 vs 65; same seeds).
- Assist on v2 (65) is below assist on legacy (77) on the same seeds. By first failed step, the spoon pick goes
  from 1 to 10 failures and the pour from 5 to 10; bottle put-back is the largest cause in both (12 legacy, 14 v2).

**Learned-policy nondeterminism: source found** (`determinism_*.json`, `scripts/policy_determinism.py`).
Inference and physics are bit-exact. With identical simulator state the front camera differs by 1 intensity level
on at most 14 of 12,288 pixels, at multisampled shadow edges: 13 of 18 run pairs diverged with default rendering,
0 of 28 with shadows off, 0 of 28 with multisampling off. `--no-msaa` renders reproducibly but is not what the
current policy was trained on; its effect on success is not measured yet.

**Learned policy on dev, ACT-Lite INT8 on iGPU, learned plate+mug picks** (`policy_dev/summary_*.md`, produced by
`scripts/policy_runs_report.py`; intervals are a cluster bootstrap over seeds):

| Rendering | Runs x seeds | Full-task success | Per run | Episodes identical in every run |
|---|---|---|---|---|
| Default (multisampled) | 3 x 100 | 35.3 % (27.0-43.7) | 34, 34, 38 | 24/100 |
| `--no-msaa` | 2 x 100 | 41.0 % (31.0-50.0) | 41, 41 | 100/100 |

- This replaces "5/10" as the learned-policy baseline. The scripted expert scores 77/100 on the same seeds.
- `--no-msaa` makes the evaluation exactly repeatable. Its difference from default rendering is +5.7 points
  (95 % CI -2.7 to +14.3, paired bootstrap over seeds): no detectable shift, even though the policy was trained
  on multisampled images.
- Weakest sub-goals (default rendering, mean of runs): bottle put-back 58 %, poured 63 %, mug placed 78 %. The
  learned picks succeed on the first attempt 75 % of the time.
- Episodes that hit the simulated-step cap are counted as failures: one in default run 3 (seed 434) and one in
  each `--no-msaa` run; none in default runs 1 and 2.

**Actuator and sensor realism** (`sim/config.yaml` profile `nominal`, values and sources in
`docs/sim_realism.md`; off by default, and the default reproduces `results/baseline/`, checked on legacy seeds 5
and 17). Scripted expert, assist, legacy, dev: 76/100 (67–83 %) under `nominal` against 77/100 without
(`realism/eval_expert_assist_legacy_nominal_dev.json`); 71 seeds succeed in both, 6 only without, 5 only with.
The expert plans from true state, so it sees only the actuator effects.

**Learned policy under `nominal`: the full-task number goes up while the policy gets worse.** INT8 on iGPU,
`--no-msaa`, dev (`realism/eval_policy_int8_plate-mug_nominal_nomsaa_dev.json`):

| | No realism (`policy_dev/..._nomsaa_r1`) | `nominal` |
|---|---|---|
| Full-task success | 41/100 | 63/100 (53–72 %) |
| Policy picks that succeed on the first attempt | 76 % of 200 | 49 % of 200 |
| Successful mug picks finished by the scripted retry | 40 of 92 | 57 of 89 |

- The policy only makes the first attempt (`planner/executor.py`, `attempts == 1`); retries are scripted. When
  the policy fails more, the expert does more of the picking, and episodes after a scripted mug pick pour more
  often. Full-task success with the policy is therefore not a measure of the policy. From here on, policy
  comparisons report first-attempt pick success alongside full-task success.
- Attribution, on the 50 seeds whose outcome flips (`realism/split/`; profiles `actuators_only` and
  `sensors_only`): full-task 14/50 with no realism, 20/50 with actuators only, 31/50 with sensors only, 36/50 with
  both. First-attempt pick success 73 %, 73 %, 61 %, 48 %. The sensor effects (which reach only the policy) cause
  most of the drop in the policy and most of the rise in the full task.

**Secondary pour metric (ballistic droplets, `pour:` in the config).** No episode passes it in any run so far:
the expert under `nominal` lands 0.14 % of droplets in the mug, and in a scratch run on seeds 0–9 the geometric
check passed 10/10 while the droplet check passed 0/10 at every outflow from 0 to 0.3 m/s. The geometric check
accepts a spout up to 4 cm off the mug axis, twice the mug's inner radius. The success definition
stays geometric; making the pour skill pass the droplet check is an open item.

**Test-split grasp trials, seeds 1000–1099, run once** (`test_split/grasp_trials_test.json`, commit 5d7cc91,
identical shake for all modes). Placed out of 100, with Wilson 95 % interval; dev in brackets.

| Object | contact@v2 | assist@v2 | assist@legacy |
|---|---|---|---|
| Plate | 99 (95–100 %) [100] | 100 (96–100 %) [100] | 100 [100] |
| Bottle | 69 (59–77 %) [68] | 89 (81–94 %) [89] | 70 (60–78 %) [87] |
| Mug | 100 (96–100 %) [99] | 100 (96–100 %) [100] | 100 [100] |
| Spoon | 0 (0–4 %) [0] | 80 (71–87 %) [76] | 96 [96] |
| Fork | 0 (0–4 %) [0] | 100 (96–100 %) [100] | 98 [97] |

- Dev and test agree within their intervals except the assist@legacy bottle (70 test vs 87 dev; its pick is 91
  vs 95, so the loss is at placing).

**Evaluation robustness.** Per-episode wall-clock budget (suspended time not charged) and simulated-step cap;
episodes that raise are recorded and counted instead of aborting the run.

**Scripted expert, full task, v2 scene, test split 1000–1099, run once** (`test_split/eval_expert_*_test.json`;
0 timeouts, 0 errors; dev in brackets):

| Grasp | Success | Main failure causes (episodes) |
|---|---|---|
| Weld assist | 66/100 (56–75 %) [65] | spoon pick 14, bottle put-back 12, pour 4 |
| Contact, fork on weld assist | 65/100 (55–74 %) [65] | pour 9, plate place 9, spoon pick 6, plate re-pick 4 |
| Contact (friction only) | 9/100 (5–16 %) [11] | fork pick 58, fork hand-over 10, plate place 9 |

**Phase 1 closing summary**
- Exit criteria met: per-object grasp trials with 100 seeds each, both modes on identical scenes, Wilson
  intervals, dev and test; scripted expert with contact grasps on the test split.
- Contact grasps hold the plate, mug and bottle; cutlery does not survive a shake and the fork pick fails in the
  full task (documented limitation). With only the fork on the weld assist, contact matches full assist (65 vs 66
  on test).
- Defaults are unchanged (`assist`, `legacy`, realism `none`), so the existing demos and baselines still apply;
  the brief's re-collection is not needed for these defaults.
- Open items: bottle placing in contact mode (68–69 %); spoon pick with assist on v2 (76–80 %); pour passes the
  geometric check but not the droplet check; STS3215 torque limit in the model above the datasheet stall torque;
  policy evaluation should report first-attempt pick success.

## Phase 2 — data engine and first fine-tune (2026-11-18 to 2026-11-20), target not met

Artifacts in `results/phase2/`. Dev seeds 400–499, `--scene v2 --grasp contact --assist-objects fork
--realism nominal --no-msaa`, picks only, FP32 PyTorch on CPU. "First attempt" counts only the policy's first
try at each pick (200 picks = 100 seeds × plate and mug).

**Built:** crash-safe shard writes and a shard validator (`scripts/validate_shards.py`), a streaming shard dataset
with a manifest (`policy/shards.py`, `policy/stream.py`), fine-tuning from a checkpoint (`--init`), per-joint
statistics recomputation (`--restat`), the recipe runner `scripts/phase2_experiments.sh`, and the per-object target
(`scripts/phase2_target.py`).

**Target** (`target.json`): non-inferiority per object to the legacy-clean level, measured on the legacy scene with
weld assist and no realism: plate 200/200 (100 %), mug 104/200 (52 %).

**Existing policy on v2 dev, first attempt** (`baseline_policy_picks_dev.md`, measured with the checkpoint's own
legacy statistics, unchanged): plate 0/100, mug 10/100, 5 % of 200 picks.

**Finding: the legacy statistics do not fit v2.** Joint 4 (arm 1 wrist roll) hardly moved in the legacy demos
(std 0.001) but the v2 expert rotates it (std 0.2). Normalising by 0.001 scales that joint about 200 times too large:
the legacy checkpoint's validation L1 on 9 v2 episodes is 11.37 (106 on that joint alone; 2.72 over the others),
against 0.0499 on its own validation data. The inflated state input is a plausible cause of much of the 5 % above;
I did not test that, because the weights cannot be evaluated fairly with other statistics. Fix for fine-tuning:
`--restat 4` keeps the checkpoint's statistics for the other 11 joints and recomputes joint 4's mean and std on the
new data (state 0.0216 / 0.206, action 0.0254 / 0.209). The patched set is stored in the new checkpoint, which
the evaluator, the runtime and the OpenVINO export already read. A 5-epoch check brought validation L1 from 11.37
to 0.118.

**Recipes** (both: `python -m policy.train --data data/v2_hybrid_nominal --aug sensor --aug-profile nominal
--init models/act_pick.pt --restat 4 --lr 1e-4 --workers 0 --buffer-shards 8 --batch 128 --seed 0 --epochs 20`):

| Recipe | Expert seeds | Train time | Plate first attempt | Mug first attempt |
|---|---|---|---|---|
| Existing policy (legacy statistics) | – | – | 0/100 | 10/100 |
| E | 81 (100–180) | 30 min | 8/100 (4–15 %) | 31/100 (23–41 %) |
| F | 101 (100–200) | 22 min | 20/100 (13–29 %) | 33/100 (25–43 %) |
| Target | – | – | 100 % | 52 % |

Both E and F are below the target for both objects beyond noise (`picks_E.json`, `picks_F.json`; 0 timeouts, 0
errors). More data moved the plate from 8 % to 20 % and the mug not at all within noise. Validation L1 is not
comparable between E and F because the held-out episodes changed (9 vs 11). Plan stopped here by decision; no
further iteration.

**Not done:** DAgger, recovery data, SmolVLA, training from scratch (recipes A–C in the runner), and the 100 seeds
of expert data beyond seed 200 (collection stopped at 101 of 300 seeds; a memory-pressure kill ended the last
collection run at seed 200, and the remaining seeds 201–210 were skipped). Hardware limit: the machine runs low on
memory, so every job here ran alone with training workers 0 and collection or evaluation workers 2.

**Scripted expert, full task, 50 dev seeds, one configuration** (hybrid contact grasp, v2, nominal realism;
`fulltask_dev/eval_expert_hybrid_nominal_dev50.json`; for the README): 22/50 = 44 % (31–58 %), 0 timeouts, 0
errors. Sub-goal failures (episodes): spoon 18, mug 14, bottle 13, pour 12, plate 10, fork 5, drawer 1. This is
below the Phase 1 dev rows (65 % with no realism); the difference is the nominal realism profile and was not
separated further.

Open items: why the plate pick stays at 8–20 % while validation L1 is about 0.1; whether recovery data or DAgger
closes it; the other statistics differences between legacy and v2 (joint stds differ by up to 2×).
