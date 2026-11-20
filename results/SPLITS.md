# Seed splits

Every scene is generated from one integer seed (`sim/scene.py::build_scene`), so a seed range is a dataset split.
The ranges live in `scripts/splits.py`; `python tasks.py eval --split …` reads them from there and refuses `train` and `perception`.

| Split | Seeds | Use | Rule |
|---|---|---|---|
| train | 100–399 | Expert demonstrations in `data/demos/` (300 episodes); any later DAgger, recovery or augmentation data | Never evaluated on |
| dev | 400–499 | Model selection, hyper-parameters, ablations, thresholds, physics tuning | The only split decisions are made on |
| test | 1000–1099 | Final numbers | Run once per reported system; never tuned on |
| legacy | 0–19 | Seeds the scripted skills were developed on (0–9) and the first held-out check (10–19) | Only to reproduce the tables of the first submission |
| perception | 20000–29999 | Rendered scenes (reset, drawer-open and perturbed layouts, no demonstrations) for the object-pose CNN, `perception/generate.py` | Never evaluated on; the checkpoint is chosen on the last 5 % of these seeds, the gate is measured on dev |

Seeds outside these ranges are unused. The perception range was added on 2026-11-20, before any data was generated in it. New data collection must stay inside 100–399 or use a new range that is
added to this file first.

## Scene versions

A seed fixes the randomization, but the same seed builds a different scene per scene version
(`sim/scene.py::SCENES`):

| Scene | What it is | Used for |
|---|---|---|
| `legacy` | The first-submission scene: flat plate, visual-only mug handle, cutlery 22 mm apart, 85 mm drawer | Reproducing `results/baseline/`; the existing demos and the current policy |
| `v2` | Rimmed plate, colliding mug handle, cutlery 38 mm apart, 115 mm drawer | Every comparison of assist and contact grasps |

Assist and contact grasps are only compared on `v2`, where objects, layout and randomization are identical for a
given seed (checked in `tests/test_contact.py`). `legacy` numbers are never placed in the same comparison.

## Notes and disclosures

- **Contact-grasp tuning used dev seeds 400–431** (pad geometry, squeeze torque, grasp points). Dev-split grasp
  numbers are therefore tuned-on numbers; the untouched measurement is the test split, run once at the end of
  Phase 1.
- **Learned-policy runs are not repeatable** with default rendering (see `results/phase1/determinism_*.json`), so
  they are reported as the mean of at least three runs of 100 seeds.
- **The training range is 100–399, not 0–299** (confirmed 2026-11-16; no re-collection). The files are `data/demos/ep_00100.npz … ep_00399.npz`, and
  `run_all.sh` and the README agree. A dev range of 300–399 would therefore have been training data, so dev is
  400–499.
- **Validation loss** in `policy/train.py` uses the first 10 % of the sorted demo files (seeds 100–129), without
  augmentation. That is a within-train holdout for the L1 loss only; it is not the dev split.
- **Prior exposure of the test range.** The scripted expert was run on seeds 1000–1029 before this split was
  written down (the "27/30" row), and that result was seen. No code was changed in response to it, and no learned
  policy has been evaluated there. To keep this visible, final test tables report 1000–1029 and 1030–1099
  separately as well as pooled.
- **Legacy seeds 0–9 are development seeds for the scripted skills.** Results on them are optimistic for the
  scripted expert. They are not training seeds for the learned policy.
- **Scene generation changes.** If a later phase changes `build_scene` (new objects, new randomization), the same
  seed produces a different scene. Splits stay valid because they are defined by seed, but results from before and
  after such a change are not comparable; each results folder records the commit it was produced at.
