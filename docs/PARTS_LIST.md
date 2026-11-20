# Parts list: perception-driven bimanual table setting

The hardware the simulated pipeline assumes (scene: `sim/scene.py`, legacy scene with weld-assist grasps;
perception: `perception/`). Each line names the simulation detail it comes from. No prices are given: none of
them would come from a repo artifact, so check current vendor listings.

## Robot

| Qty | Part | Specification | Where the simulation uses it |
|---|---|---|---|
| 2 | SO-101 arm (follower), assembled with its servos | 5-DoF arm + 1-DoF gripper, position-controlled bus servos | `sim/scene.py` arms A (base at y=+0.17 m) and B (y=-0.17 m), 0.34 m apart |
| 2 | Servo bus controller board for the SO-101 | USB, one per arm | Joint targets at 20 Hz (`sim/env.py` control rate) |
| 2 | Power supply for each arm | As specified by the SO-101 build guide for its servo variant | — |
| 1 | Rigid table or base plate | Both arm bases bolted at known positions; the planner uses fixed base poses | `planner/skills.py` uses each arm's base position directly |
| — | Clamps / bolts for the arm bases | Bases must not move: the held-bottle pose is forward kinematics plus a fixed grasp offset | `perception/grasp_offsets.json` |

## Perception

| Qty | Part | Specification | Where the simulation uses it |
|---|---|---|---|
| 2 | USB camera | ≥ 320×240 colour, fixed exposure; the CNN input is 240×320 per camera | `perception/model.py`, `perception/labels.py` (`CAMERAS`) |
| 1 | Overhead camera mount | Camera at (0.22, 0, 0.75) m in the table frame, looking straight down, about 55° vertical FOV | `sim/scene.py` camera `overhead` |
| 1 | Front camera mount | Camera at (0.75, 0, 0.45) m, looking back at the table, pitched down about 30°, about 50° vertical FOV | `sim/scene.py` camera `front` |
| 1 | Camera calibration target | Checkerboard or ChArUco board, to place both cameras in the table frame | The CNN was trained on these exact viewpoints |
| 1 | Computer with an NVIDIA GPU | Runs the pose CNN: 23.5 ms median and 29 ms p95 per two-camera estimate, about 5 estimates per episode, measured in the simulator on this project's laptop GPU (`results/perception/eval_perception_calibrated_setdown_legacy_dev.json`, `latency_ms`) | `perception/estimator.py` |

## Task objects and fixtures

| Qty | Part | Specification | Where the simulation uses it |
|---|---|---|---|
| 1 | Small cabinet with a sliding drawer and a bar handle | Handle at (0.245, 0, 0.072) m when shut; the task needs it open more than 60 mm (`drawer_open`) | `perception/estimator.py` `HANDLE_CLOSED`; `sim/scene.py` |
| 1 | Plate | Flat, about 12 mm thick | `PLATE_Z` |
| 1 | Mug with a handle | Cylindrical or box-shaped (both are randomised) | `sim/env.py` randomisation |
| 1 | Bottle | Graspable below the cap, tall and narrow | `sim/scene.py` |
| 1 each | Fork and spoon | Start inside the drawer | `sim/scene.py` |

## What the simulation does not yet cover

These are the gaps between the simulated result and a hardware run. Each needs its own test before the numbers above
can be expected on real hardware.

- **Grasping.** The headline numbers use weld-assist grasps: an object locks to the gripper when the jaws close
  within 3 cm of its grasp point (`sim/env.py`). Real SO-101 jaws must hold the bottle, the mug and the cutlery by
  friction.
- **Knowing whether a gripper holds something.** The executor reads `env.held`. On hardware this has to come from the
  gripper servo's position or current when it closes.
- **The task checks** (`env.check`, used for retries and repairs) read the simulator state. On hardware they have to
  come from the cameras.
- **The held-bottle grasp offset** was calibrated in simulation on training seeds 100–119. Calibrate it again on
  the real gripper and bottle.
- **Rendering and appearance.** The CNN was trained on MuJoCo renders only (`--realism none`).

The set-down pour also changes the task: arm B no longer holds the mug while arm A pours. It sets the mug at the
pour spot, moves clear, and re-grasps it afterwards.
