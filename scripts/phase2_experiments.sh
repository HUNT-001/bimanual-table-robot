#!/usr/bin/env bash
# Phase 2 first batch: which training recipe reaches the per-object target on v2 dev with the hybrid grasp and
# nominal realism? Each step is skipped when its output exists, so an interrupted run can simply be started again.
#
#   bash scripts/phase2_experiments.sh A B          # expert data only (needs data/v2_hybrid_nominal)
#   bash scripts/phase2_experiments.sh C D          # adds the recovery data (data/v2_hybrid_nominal_dart)
#
# A  scratch, expert data, original augmentation (brightness jitter + state noise)
# B  scratch, expert data, sensor augmentation mirroring the nominal realism profile
# C  scratch, expert + recovery data, sensor augmentation
# D  fine-tune from the legacy checkpoint (models/act_pick.pt), expert + recovery data, sensor augmentation
# All train with seed 0, batch 128; evaluation is picks-only on dev seeds 400-499, --no-msaa, FP32 torch on CPU.
# Compare with: python scripts/phase2_target.py --candidate results/phase2/picks_<X>.json
set -u
cd "$(dirname "$0")/.."
mkdir -p models/phase2 results/phase2/logs
EXPERT=data/v2_hybrid_nominal
DART=data/v2_hybrid_nominal_dart
COMMON="--workers 2 --buffer-shards 8 --batch 128 --seed 0"
SENSOR="--aug sensor --aug-profile nominal"

recipe() {
  case "$1" in
    A) echo "--data $EXPERT --aug legacy --epochs 60" ;;
    B) echo "--data $EXPERT $SENSOR --epochs 60" ;;
    C) echo "--data $EXPERT --data $DART $SENSOR --epochs 60" ;;
    D) echo "--data $EXPERT --data $DART $SENSOR --init models/act_pick.pt --lr 1e-4 --epochs 20" ;;
    *) echo "unknown recipe $1" >&2; return 1 ;;
  esac
}

for X in "$@"; do
  if [ ! -f models/phase2/$X.pt ]; then
    python -m policy.train $COMMON $(recipe $X) --out models/phase2/$X.pt > results/phase2/logs/train_$X.log 2>&1
    echo "train $X exit $?"
  fi
  if [ ! -f results/phase2/picks_$X.json ]; then
    OMP_NUM_THREADS=2 python scripts/evaluate.py --seeds 400-499 --workers 2 --scene v2 --grasp contact \
      --assist-objects fork --realism nominal --no-msaa --policy models/phase2/$X.pt --device CPU \
      --stop-after-policy --out results/phase2/picks_$X.json > results/phase2/logs/picks_$X.log 2>&1
    echo "eval $X exit $?"
  fi
done
