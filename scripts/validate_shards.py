"""Validate a shard directory against its manifest; optionally delete files the manifest does not list.

    python scripts/validate_shards.py data/v2_hybrid_nominal                    # report, exit 1 on any problem
    python scripts/validate_shards.py data/v2_hybrid_nominal --delete-orphans   # first remove unlisted files
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from policy import shards as S  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--delete-orphans", action="store_true",
                    help="delete files in shards/ that have no manifest row (an interrupted write)")
    a = ap.parse_args()
    if a.delete_orphans:
        for f in S.remove_orphans(a.root):
            print(f"deleted unlisted file: {f}")
    rows = S.load_manifest(a.root)
    problems = S.validate(a.root)
    print(f"{a.root}: {len(rows)} manifest rows, {sum(r['n_frames'] for r in rows)} frames, {len(problems)} problems")
    for p in problems:
        print("  PROBLEM:", p)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
