#!/usr/bin/env bash
# End-to-end reproduction. GPU recommended for step 2; steps 4-6 must run on the Intel target.
set -e
SEEDS_EVAL=${SEEDS_EVAL:-0-9}
DEMO_SEEDS=${DEMO_SEEDS:-100-399}
DEV=${DEV:-GPU}   # OpenVINO device for the learned-policy evaluation (CPU | GPU | NPU)

# 1) collect expert demonstrations for the learned pick skill
python -m policy.collect --seeds "$DEMO_SEEDS" --skills pick --out data/demos
# 2) train ACT-Lite
python -m policy.train --data data/demos --out models/act_pick.pt --epochs 60
# 3) export to OpenVINO IR (FP32 / FP16 / INT8)
python -m deploy.export_openvino --ckpt models/act_pick.pt --data data/demos --out models
# 4) Intel benchmark
python -m deploy.benchmark --models models/act_pick_fp32.xml models/act_pick_fp16.xml models/act_pick_int8.xml \
    --torch models/act_pick.pt --sim --out results/benchmark.json
# 5) 10-seed evaluation: scripted expert and learned policy (+ videos)
python scripts/evaluate.py --seeds "$SEEDS_EVAL" --out results/eval_expert.json --videos videos
python scripts/evaluate.py --seeds "$SEEDS_EVAL" --policy models/act_pick_int8.xml --device "$DEV" \
    --out results/eval_policy_int8.json
# 6) stitched submission video
python scripts/make_submission_video.py --videos videos --eval results/eval_expert.json --out videos/submission.mp4
