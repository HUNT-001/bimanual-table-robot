# Simulation realism: what is modelled, with what values, and from where

Parameters live in `sim/config.yaml` (`realism:` profiles). The default profile `none` is the idealised simulator
and reproduces `results/baseline/`. `--realism nominal` (`scripts/evaluate.py`) switches the effects below on.
Nothing here has been identified against a physical SO-101; "assumed" marks values with no source beyond judgement.
Looked up on 2026-11-17; verify against the documentation of the exact kit before relying on any of it.

## Actuators

| Effect | Value in `nominal` | How it is modelled | Source |
|---|---|---|---|
| Position gain | kp 998.22, kv 2.731 | MuJoCo position actuator per joint | Shipped in `sim/assets/so101/so101.xml` (TheRobotStudio SO-ARM100 model); its comment says the gains are not a 1:1 mapping of the servo gains and were derived assuming servo P gain 16 |
| Torque limit | 3.35 N·m on every joint | actuator `forcerange` | Same model file. The published peak stall torque of the 12 V STS3215 is 30 kg·cm = 2.94 N·m and of the 7.4 V part 19.5 kg·cm = 1.91 N·m [1][2], so **the model's limit is above the datasheet stall torque**. Not changed here, to keep the baseline; flagged for system identification |
| Velocity limit | 4.7 rad/s | slew limit on the commanded target per control step | No-load speed 0.222 s/60° at 12 V [2] = 4.72 rad/s. Under load the real limit is lower (assumed not modelled) |
| Deadband | 0.0015 rad | target changes below this are not sent | One encoder count: 4096 counts per revolution [1][3] = 0.00153 rad. The servo's own deadband setting is configurable; one count is assumed |
| Backlash | 0.0087 rad (0.5°) total play | the output follows the target only after the play is taken up | `so101.xml` defines a `backlash` joint class commented "+/- 0.5° of backlash". Our model applies 0.5° total at the command level, not as a separate joint (assumed simplification) |
| Control latency | 1–3 control steps (50–150 ms), fixed per episode | the servos track the command issued that many steps earlier | Assumed. Serial bus round trip, camera capture and inference on a laptop; no measurement behind it |
| Joint friction, damping, armature | damping 0.60, frictionloss 0.052, armature 0.028 | MuJoCo joint properties | Shipped in `so101.xml` |

## Sensors

| Effect | Value in `nominal` | How it is modelled | Source |
|---|---|---|---|
| Joint-reading noise | 0.002 rad std | Gaussian noise on the joint vector returned to the policy | Assumed; about 1.3 encoder counts |
| Camera noise | 2 intensity levels std | per-pixel Gaussian noise | Assumed (typical of a small USB camera in good light) |
| Camera blur | 3×3 box blur on 20 % of frames | applied after rendering | Assumed (motion/focus blur) |
| Camera exposure | gain 0.9–1.1, fixed per episode | multiplies the image | Assumed |
| Camera extrinsics | ±5 mm per axis, ±2° per axis, fixed per episode | perturbs the policy cameras' pose | Ranges requested in the project brief; representative of remounting a camera by hand |

Scripted skills still plan from the true simulator state (object poses, true joint angles). Sensor effects therefore
reach the learned policy only; actuator effects reach everything.

## Grasping and pouring

- **Grasp assist** (`grasp="assist"`): a weld constraint. Not physical.
- **Contact grasp** (`grasp="contact"`): friction only, with box fingers and pads replacing the convex-hull finger
  meshes. The gripper squeezes at the model's 3.35 N·m limit, which (see above) exceeds the published stall torque
  and could not be held continuously on hardware. Pad friction (1.6 sliding, 0.03 torsional) is assumed for a
  rubber pad and is not measured. The fork cannot be held this way in our trials.
- **Pouring** is judged geometrically (bottle tilted more than 55° with its spout within 4 cm above the mug for
  0.8 s). There is no fluid.

## Not modelled

Servo temperature and voltage sag, torque–speed curve, cable drag, encoder quantisation of the reading, rolling
shutter, lens distortion, depth, and any object property beyond rigid shape, mass and friction.

## Sources

1. Feetech STS3215 datasheet (translated), Core Electronics: https://core-electronics.com.au/attachments/uploads/sts3215-smart-servo-datasheet-translated.pdf
2. FeeTech 12V 30kg.cm Magnetic Encoding Servo STS3215, RobotShop listing: https://ca.robotshop.com/products/feetech-12v-30kgcm-magnetic-encoding-servo-sts3215
3. Feetech STS3215-C044 listing (12-bit magnetic sensor), OpenELAB: https://openelab.io/products/feetech-sts3215-c044-servo-7
4. SO-ARM101 kit overview (follower: six 1:345 motors; 12 V Pro kit or 7.4 V standard kit), CNX Software: https://www.cnx-software.com/2025/05/02/so-arm101-open-source-dual-robotic-arm-kit-works-with-hugging-faces-lerobot/
5. Seeed Studio SO-ARM course, motor configuration and power supplies: https://wiki.seeedstudio.com/robotics/soarm_physical_ai_course_chapter_6/

Retail listings disagree on part codes and some figures; the datasheet for the exact part in the kit takes precedence.
