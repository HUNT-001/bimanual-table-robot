# Bimanual Dinner-Table Setting — Dual SO-101 · MuJoCo · OpenVINO

Intel Physical AI Online Challenge — *Bimanual VLA Manipulation with Multi-Modal Reasoning*
(challenge option: **Setting up a dinner table**).

Two simulated SO-101 arms take a natural-language command and complete the whole task in a
randomized MuJoCo scene: open a drawer, fetch cutlery, **hand a fork from one arm to the other**,
set the plate, and **pour from a bottle into a mug that the other arm holds steady**. The learned
visuomotor policy (ACT-Lite, language-conditioned) is exported to OpenVINO IR (FP32 / FP16 / INT8)
and benchmarked on Intel CPU, iGPU and NPU.

```
"Open the top drawer with arm A, pick up the plate with arm A and place it on the table, take the spoon
 with arm B …, pick up the fork with arm B and hand it over to arm A, … pour water into the mug with arm A …"
```

## Results (10 randomized seeds)

| | |
|---|---|
| Full-task success, evaluation seeds 0–9 (all 7 sub-goals, scripted-expert executor) | **10 / 10 (100 %)** |
| Full-task success, held-out seeds 10–19 (never used while tuning) | **7 / 10 (70 %)** — per-sub-goal ≥ 80 % |
| Sub-goals checked | drawer open · plate / spoon / fork / mug / bottle on target (upright) · poured |
| Randomized per seed | object positions, masses (mug 30–120 g, bottle 40–100 g, plate 50–150 g), friction, object size (±15 %), mug shape (cylinder/box), lighting, sky/floor/table colours |

Seeds 0–9 were used while developing the skills, so the held-out 70 % is the fairer robustness number
(failures: drawer pushed shut / bottle or spoon not placed after a retry). Tables: `results/eval_expert*.md`.

Regenerate: `python scripts/evaluate.py --seeds 0-9 --out results/eval_expert.json --videos videos/`.
Learned-policy results (`results/eval_policy_int8.md`) report **policy-only success** (step solved on the
first try, no expert fallback) separately from task success.

## Architecture

```
 NL command ──► Planner (rule grammar | OpenVINO-GenAI LLM, same JSON schema)
                  │  plan: [open_drawer(A), pick(A,plate), place(A,plate), pick(B,spoon) …
                  │         handoff(B→A,fork), pick(B,mug), pick(A,bottle), pour(A,B), …]
                  ▼
 Closed-loop Executor ── per step: execute → re-observe → check post-condition → retry / re-plan
   │        │
   │        ├─ Learned skill: ACT-Lite (2 cams + 12-D state + sub-instruction tokens → 20-step action chunk)
   │        │     OpenVINO IR on Intel CPU / iGPU / NPU, temporal ensembling
   │        └─ Scripted expert skills (also the demonstrator): IK, object-space carrying, hand-over, pour
   ▼
 MuJoCo: two SO-101 (TheRobotStudio MJCF, STS3215 servo model), drawer cabinet, plate, mug, bottle, cutlery
```

| Folder | What |
|---|---|
| `sim/scene.py` | Procedural MjSpec scene + domain randomization (seeded) |
| `sim/env.py` | Env: 20 Hz control, cameras, damped-least-squares IK, tool-point IK, task checks |
| `planner/language.py` | Instruction → plan (clause grammar with holding-context, pronouns, arm references; optional LLM) |
| `planner/skills.py` | Bimanual skill library (pick, place, open_drawer, handoff, pour, carry) |
| `planner/executor.py` | Closed-loop executor with post-condition checks and recovery |
| `policy/` | ACT-Lite model, demo collection, training, runtime |
| `deploy/` | OpenVINO export (FP32/FP16/INT8 via NNCF) and Intel benchmark |
| `scripts/` | Episode runner, 10-seed evaluation, annotated video writer, submission video |

### Bimanual coordination
* **Hand-over (fork):** arm B grasps the fork by its own-side end and carries it to a mid-line meeting
  point; arm A aligns its wrist roll with the fork axis and grasps the free end; A closes first, then B
  releases (attachment transfers); B retreats.
* **Complementary action (pour):** arm B slides the mug to the pouring spot and **turns it** (wrist roll
  search) so the handle is perpendicular to arm A's reach; B keeps holding it while arm A carries the
  bottle through a collision-free corridor and tilts it in stages (40°→90°) with tool-point IK on the
  spout, entering high and from its own side so the bottle never intersects the mug.
* **Shared-workspace sequencing:** only one arm moves at a time inside the shared zone; carried tall
  objects use raised, densified Cartesian paths; a friction detent keeps the drawer open.

### Multi-modal reasoning
* The planner keeps a *holding context* across clauses ("place **it**", implicit pickers for a
  hand-over, auto-inserts `open_drawer` when cutlery is requested, puts down anything still held).
* The executor verifies every sub-goal against the observed scene and retries or re-grasps if an object
  was dropped, i.e. the plan adapts to the scene state.
* The learned policy consumes both camera images, the joint state and the sub-instruction.

### Robustness
Randomization of placement, mass, friction, size, shape, lighting and background every seed;
closed-loop post-condition checks with recovery; forward-kinematics-corrected placement (held-object
offsets rotate with the wrist); IK restricted to one elbow branch for consistent motion.

### Training (ACT-Lite)
ACT without the CVAE branch: 4-stage conv backbone per camera → tokens, + state token + instruction
tokens → 3-layer transformer encoder → 2-layer decoder with 20 action queries → 12-D joint targets.
~1.6 M parameters. L1 loss, AdamW + OneCycle, image-intensity and state-noise augmentation.
Demos come from the scripted expert on seeds 100+ (disjoint from evaluation seeds 0–9).

```bash
python -m policy.collect --seeds 100-399 --skills pick --out data/demos     # ~any CPU, parallel
python -m policy.train   --data data/demos --out models/act_pick.pt --epochs 60   # GPU recommended
```

### OpenVINO optimization & Intel hardware mapping
```bash
python -m deploy.export_openvino --ckpt models/act_pick.pt --data data/demos --out models
python -m deploy.benchmark --models models/act_pick_{fp32,fp16,int8}.xml --torch models/act_pick.pt --sim
```
* TorchScript trace → `ov.convert_model` → IR; FP16 weight compression; INT8 post-training
  quantization with NNCF (`ModelType.TRANSFORMER`, mixed preset, 300 calibration frames from demos);
  parity check vs PyTorch printed at export.
* Benchmark reports mean / p50 / p90 / p99 latency, sync FPS, async throughput (AUTO number of
  infer requests), compile time, device name and precision for every available device.

| Workload | Intel target |
|---|---|
| MuJoCo physics + rendering | CPU P-cores (+ iGPU for OpenGL rendering) |
| ACT-Lite policy (20 Hz, replan every 5 steps) | iGPU (FP16/INT8) or NPU (INT8); CPU fallback |
| Optional LLM planner (openvino-genai, INT4) | iGPU / NPU, runs once per command |

## Quick start
```bash
pip install -r requirements.txt
python scripts/run_episode.py --seed 3 --video videos/seed3.mp4        # one annotated episode
python scripts/evaluate.py --seeds 0-9 --videos videos/                  # 10-seed evaluation + videos
./run_all.sh                                                             # full pipeline
```
Linux headless rendering uses `MUJOCO_GL=egl` (set automatically); on Windows it uses the default WGL.

## Honest notes / limitations
* **Grasp assist.** When a gripper closes on the *intended* object within 3 cm, a weld constraint is
  activated (and transferred during hand-overs). Pinch grasps with the SO-101 mesh jaws are brittle in
  MuJoCo; the assist keeps the evaluation on sequencing, coordination and perception. Arms, objects,
  drawer, table and each other still collide physically.
* **Pouring is geometric** (no fluid simulation): counted when the bottle is tilted > 55° with its spout
  above the mug rim for ≥ 0.8 s.
* **Learned vs scripted.** The full task is completed by the closed-loop executor with scripted skills;
  the learned ACT-Lite policy currently covers the *pick* skill (with expert fallback). Results are
  reported separately so it is clear what the policy itself achieves.
* **Hardware.** Development benchmark numbers in this repo come from a non-Core-Ultra machine; the
  benchmark script auto-detects and uses NPU/iGPU when run on Core Ultra Series 2/3.

## Credits
SO-101 model: TheRobotStudio/SO-ARM100 (Apache-2.0). MuJoCo (DeepMind), OpenVINO & NNCF (Intel),
ACT (Zhao et al., 2023).
