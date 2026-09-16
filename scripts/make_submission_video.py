"""Stitch per-seed videos into one submission video with title cards and a results summary.

    python scripts/make_submission_video.py --videos videos/ --eval results/eval_expert.json --out videos/submission.mp4
"""
import argparse
import glob
import json
import os

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from video import _font  # noqa: E402  (same folder)


def card(size, lines, sizes=None):
    img = Image.new("RGB", size, (14, 16, 24))
    d = ImageDraw.Draw(img)
    y = size[1] // 2 - 20 * len(lines)
    for i, l in enumerate(lines):
        f = _font((sizes or [30] + [20] * 20)[i])
        w = d.textlength(l, font=f)
        d.text(((size[0] - w) / 2, y), l, font=f, fill=(235, 235, 235) if i else (120, 200, 255))
        y += 44 if i == 0 else 30
    return np.asarray(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default="videos")
    ap.add_argument("--eval", default="results/eval_expert.json")
    ap.add_argument("--bench", default="results/benchmark.json")
    ap.add_argument("--out", default="videos/submission.mp4")
    ap.add_argument("--fps", type=int, default=10)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.videos, "seed_*.mp4")))
    assert files, "no seed videos found"
    ev = json.load(open(a.eval)) if os.path.exists(a.eval) else None
    first = imageio.get_reader(files[0]).get_next_data()
    size = (first.shape[1], first.shape[0])
    w = imageio.get_writer(a.out, fps=a.fps, codec="libx264", quality=7, macro_block_size=1)
    intro = ["Bimanual VLA-style Dinner-Table Setting", "Dual SO-101 arms | MuJoCo | OpenVINO on Intel",
             "Language -> plan -> closed-loop bimanual skills", f"{len(files)} randomized seeds"]
    for _ in range(a.fps * 3):
        w.append_data(card(size, intro))
    for f in files:
        seed = int(os.path.basename(f)[5:7])
        ep = next((e for e in ev["episodes"] if e["seed"] == seed), None) if ev else None
        rz = ep["randomization"] if ep else {}
        lines = [f"Seed {seed}",
                 f"mug: {rz.get('mug_shape', '?')} {rz.get('mug_g', '?')} g | bottle {rz.get('bottle_g', '?')} g | "
                 f"plate {rz.get('plate_g', '?')} g",
                 f"light {rz.get('light', '?')} | table friction {rz.get('table_mu', '?')} | size x{rz.get('size_scale', '?')}"]
        for _ in range(int(a.fps * 1.5)):
            w.append_data(card(size, lines))
        for fr in imageio.get_reader(f):
            w.append_data(fr)
    if ev:
        s = ev["summary"]
        lines = ["Results", f"task success: {s['success_rate'] * 100:.0f}% over {s['n']} seeds"]
        lines += [f"{k}: {v * 100:.0f}%" for k, v in s["subgoal_rates"].items()]
        if os.path.exists(a.bench):
            b = json.load(open(a.bench))
            for r in b["results"]:
                if "error" not in r:
                    lines.append(f"{r['runtime']} {r['device']} {r['precision']}: {r['mean_ms']:.1f} ms")
        for _ in range(a.fps * 6):
            w.append_data(card(size, lines))
    w.close()
    print("wrote", a.out)


if __name__ == "__main__":
    main()
