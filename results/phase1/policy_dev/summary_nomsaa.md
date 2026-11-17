# Learned plate+mug picks, ACT-Lite INT8 on iGPU, dev seeds, multisampling off

2 runs x 100 seeds (400-499); config: {'policy': 'models/act_pick_int8.xml', 'device': 'GPU', 'policy_objects': 'plate,mug', 'grasp': 'assist', 'scene': 'legacy', 'msaa': False}

**Full-task success: 41.0 %** (95 % CI 31.0-50.0, cluster bootstrap over seeds); std across runs 0.0 points

| Run | Successes | Wilson 95 % | Timed out | Errors |
|---|---|---|---|---|
| eval_policy_int8_plate-mug_dev_nomsaa_r1.json | 41/100 | 32-51 % | 1 | 0 |
| eval_policy_int8_plate-mug_dev_nomsaa_r2.json | 41/100 | 32-51 % | 1 | 0 |

Per seed across runs: always succeeded 41, never 59, mixed 0; episodes identical step for step in every run: 100/100.

Sub-goal rates (mean over runs): bottle_placed 58 %, drawer_open 97 %, fork_placed 97 %, mug_placed 76 %, plate_placed 95 %, poured 64 %, spoon_placed 94 %

Learned steps solved on the first attempt, no scripted fallback: 76 % (mean over runs; [200, 200] steps); mean inference 3.7 ms.
