# Evaluation: models/act_pick_int8.xml  (10 seeds)  - learned picks: plate,mug

**Task success: 50%** (95% CI 24-76%)  |  mean retries 4.5  |  mean episode 131 s (sim)  |  learned-policy steps solved without fallback: 80% (20 steps, 5.3 ms/inference)

| seed | success | bottle_placed | drawer_open | fork_placed | mug_placed | plate_placed | poured | spoon_placed | retries | mug | light | table mu |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | yes | ok | ok | ok | ok | ok | ok | ok | 0 | box 92 g | 0.87 | 1.11 |
| 1 | no | x | ok | ok | ok | ok | ok | ok | 4 | cylinder 44 g | 0.41 | 0.8 |
| 2 | yes | ok | ok | ok | ok | ok | ok | ok | 1 | box 40 g | 0.73 | 0.86 |
| 3 | yes | ok | ok | ok | ok | ok | ok | ok | 0 | box 33 g | 0.46 | 0.86 |
| 4 | no | x | ok | ok | x | ok | x | x | 19 | cylinder 61 g | 0.67 | 0.86 |
| 5 | yes | ok | ok | ok | ok | ok | ok | ok | 3 | cylinder 108 g | 0.9 | 0.86 |
| 6 | no | x | ok | ok | ok | ok | ok | ok | 5 | cylinder 43 g | 0.46 | 0.61 |
| 7 | no | x | ok | ok | ok | ok | x | ok | 6 | cylinder 87 g | 0.63 | 0.75 |
| 8 | no | x | ok | ok | ok | ok | x | ok | 6 | cylinder 86 g | 0.64 | 0.71 |
| 9 | yes | ok | ok | ok | ok | ok | ok | ok | 1 | box 113 g | 0.62 | 0.6 |
| **rate** | **50%** | 50% | 100% | 100% | 90% | 100% | 70% | 90% | | | | |
