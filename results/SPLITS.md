# Seed splits

Every scene is generated from one integer seed (`sim/scene.py::build_scene`), so a seed range is a dataset split.
The ranges live in `scripts/splits.py`; `python tasks.py eval --split …` reads them from there and refuses `train`.

| Split | Seeds | Use | Rule |
|---|---|---|---|
| train | 100–399 | Expert demonstrations in `data/demos/` (300 episodes); any later DAgger, recovery or augmentation data | Never evaluated on |
| dev | 400–499 | Model selection, hyper-parameters, ablations, thresholds, physics tuning | The only split decisions are made on |
| test | 1000–1099 | Final numbers | Run once per reported system; never tuned on |
| legacy | 0–19 | Seeds the scripted skills were developed on (0–9) and the first held-out check (10–19) | Only to reproduce the tables of the first submission |

Seeds outside these ranges are unused. New data collection must stay inside 100–399 or use a new range that is
added to this file first.

## Notes and disclosures

- **The training range is 100–399, not 0–299.** The files are `data/demos/ep_00100.npz … ep_00399.npz`, and
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
