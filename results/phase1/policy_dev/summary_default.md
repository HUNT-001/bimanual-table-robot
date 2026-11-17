# Learned plate+mug picks, ACT-Lite INT8 on iGPU, dev seeds, default rendering

3 runs x 100 seeds (400-499); config: {'policy': 'models/act_pick_int8.xml', 'device': 'GPU', 'policy_objects': 'plate,mug', 'grasp': 'assist', 'scene': 'legacy', 'msaa': True}

**Full-task success: 35.3 %** (95 % CI 27.0-43.7, cluster bootstrap over seeds); std across runs 2.3 points

| Run | Successes | Wilson 95 % | Timed out | Errors |
|---|---|---|---|---|
| eval_policy_int8_plate-mug_dev_r1.json | 34/100 | 25-44 % | 0 | 0 |
| eval_policy_int8_plate-mug_dev_r2.json | 34/100 | 25-44 % | 0 | 0 |
| eval_policy_int8_plate-mug_dev_r3.json | 38/100 | 29-48 % | 1 | 0 |

Per seed across runs: always succeeded 27, never 54, mixed 19; episodes identical step for step in every run: 24/100.

Sub-goal rates (mean over runs): bottle_placed 58 %, drawer_open 99 %, fork_placed 99 %, mug_placed 78 %, plate_placed 97 %, poured 63 %, spoon_placed 97 %

Learned steps solved on the first attempt, no scripted fallback: 75 % (mean over runs; [200, 200, 200] steps); mean inference 4.0 ms.
