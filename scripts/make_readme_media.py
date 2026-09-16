"""Render the images used in the README (scene shots, key moments, randomization grid, charts, GIF).

    python scripts/make_readme_media.py --out docs/images            # everything
    python scripts/make_readme_media.py --out docs/images --charts-only
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def free_cam(lookat, dist, az, el):
    import mujoco
    c = mujoco.MjvCamera()
    c.lookat[:] = lookat
    c.distance, c.azimuth, c.elevation = dist, az, el
    return c


def render(env, cam, hw=(540, 720)):
    import mujoco
    key = ("free",) + tuple(hw)
    if not hasattr(env, "_media_r"):
        env._media_r = {}
    if key not in env._media_r:
        env._media_r[key] = mujoco.Renderer(env.model, *hw)
    r = env._media_r[key]
    r.update_scene(env.data, cam)
    return r.render().copy()


def save(img, path):
    from PIL import Image
    Image.fromarray(img).save(path, optimize=True)
    print("wrote", path)


def scene_shots(out, seed):
    """Run one full episode and grab the key moments."""
    from PIL import Image
    from planner.executor import Executor
    from planner.language import DEFAULT_INSTRUCTION, RuleParser
    from sim.env import DinnerTableEnv
    env = DinnerTableEnv(seed)
    save(render(env, "demo"), os.path.join(out, "scene_start.png"))
    save(render(env, free_cam([0.2, 0, 0.03], 0.75, 150, -35)), os.path.join(out, "scene_overview.png"))
    state = dict(handoff=False, pour=False, drawer=False, n=0)
    gif = []

    def cb(e, text):
        state["n"] += 1
        if state["n"] % 10 == 0:
            gif.append(Image.fromarray(render(e, "demo", (270, 360))))
        if not state["drawer"] and "drawer" in text and e.drawer_open() > 0.07:
            state["drawer"] = True
            save(render(e, free_cam([0.2, 0.03, 0.04], 0.5, 125, -40)), os.path.join(out, "moment_drawer.png"))
        if not state["handoff"] and "hand the fork" in text and e.held["A"] == "fork":
            state["handoff"] = True
            save(render(e, free_cam([0.13, 0.0, 0.08], 0.42, 180, -62)), os.path.join(out, "moment_handoff.png"))
        if not state["pour"] and e.poured > 0.5:
            state["pour"] = True
            save(render(e, free_cam([0.15, -0.03, 0.09], 0.40, 160, -20)), os.path.join(out, "moment_pour.png"))

    ex = Executor(env, frame_cb=cb, verbose=False)
    res = ex.run(RuleParser().parse(DEFAULT_INSTRUCTION), DEFAULT_INSTRUCTION)
    save(render(env, "demo"), os.path.join(out, "scene_final.png"))
    save(render(env, "overhead", (540, 720)), os.path.join(out, "scene_final_top.png"))
    print("episode success:", res.success, res.subgoals)
    if gif:
        gif[0].save(os.path.join(out, "episode.gif"), save_all=True, append_images=gif[1:], duration=120, loop=0,
                    optimize=True)
        print("wrote episode.gif", len(gif), "frames")
    env.close()


def randomization_grid(out, seeds=(0, 1, 2, 3, 4, 5)):
    from PIL import Image, ImageDraw
    from sim.env import DinnerTableEnv
    tiles = []
    for s in seeds:
        env = DinnerTableEnv(s)
        img = Image.fromarray(render(env, "demo", (270, 360)))
        p = env.info.params
        d = ImageDraw.Draw(img)
        label = f"seed {s} | mug {'box' if p['mug_shape'] else 'cyl'} {p['mug_mass'] * 1000:.0f} g | light {p['light_intensity']:.2f}"
        d.rectangle([0, 0, 360, 18], fill=(0, 0, 0))
        d.text((5, 3), label, fill=(255, 255, 255))
        tiles.append(img)
        env.close()
    grid = Image.new("RGB", (360 * 3, 270 * 2))
    for i, t in enumerate(tiles):
        grid.paste(t, ((i % 3) * 360, (i // 3) * 270))
    grid.save(os.path.join(out, "randomization_grid.png"), optimize=True)
    print("wrote randomization_grid.png")


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def charts(out, bench_path, evals):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch  # noqa: F401

    # ---- benchmark: latency per runtime/device/precision (single measure -> one hue)
    if os.path.exists(bench_path):
        b = json.load(open(bench_path))
        names_hw = b.get("hardware", {}).get("device_names", {})
        rows = [r for r in b["results"] if "error" not in r and
                (r["runtime"] == "pytorch" or "intel" in names_hw.get(r["device"], "intel").lower())]
        lab = {"CPU": "CPU", "GPU.0": "iGPU", "GPU": "iGPU", "NPU": "NPU"}
        names = [("PyTorch" if r["runtime"] == "pytorch" else "OpenVINO") + f" · {lab.get(r['device'], r['device'])} · "
                 f"{r['precision'].upper()}" for r in rows]
        vals = [r["mean_ms"] for r in rows]
        order = np.argsort(vals)[::-1]
        fig, ax = plt.subplots(figsize=(8, 0.42 * len(rows) + 1.2), dpi=150)
        fig.patch.set_facecolor(SURFACE)
        _style(ax)
        best = min(vals)
        for i, k in enumerate(order):
            is_pt = rows[k]["runtime"] == "pytorch"
            col = "#9c9a92" if is_pt else BLUE
            ax.barh(i, vals[k], height=0.62, color=col, edgecolor=SURFACE, linewidth=2)
            txt = f"{vals[k]:.2f} ms" + (f"  ({vals[order[0]] / vals[k]:.1f}×)" if not is_pt else "  (baseline)")
            ax.text(vals[k] + max(vals) * 0.01, i, txt, va="center", fontsize=8.5,
                    color=INK if vals[k] == best else INK2, fontweight="bold" if vals[k] == best else "normal")
        ax.set_yticks(range(len(rows)), [names[k] for k in order], fontsize=8.5, color=INK)
        ax.set_xlim(0, max(vals) * 1.35)
        ax.set_xlabel("mean latency per inference (ms, batch 1) — lower is better", color=INK2, fontsize=8.5)
        cpu = b["hardware"]["device_names"].get("CPU", "")
        ax.set_title(f"ACT-Lite policy inference — {cpu}", loc="left", color=INK, fontsize=11, pad=10)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "benchmark_latency.png"), facecolor=SURFACE)
        plt.close(fig)
        print("wrote benchmark_latency.png")

    # ---- sub-goal success rates: up to 3 executors (grouped bars, fixed categorical order)
    series = [(n, json.load(open(p))) for n, p in evals if os.path.exists(p)]
    if series:
        goals = ["drawer_open", "plate_placed", "spoon_placed", "fork_placed", "poured", "bottle_placed",
                 "mug_placed"]
        pretty = ["drawer\nopen", "plate", "spoon", "fork\n(hand-over)", "pour\n(two-arm)", "bottle\nback",
                  "mug"]
        cols = [BLUE, ORANGE, AQUA]
        fig, ax = plt.subplots(figsize=(9, 3.8), dpi=150)
        fig.patch.set_facecolor(SURFACE)
        ax.set_facecolor(SURFACE)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=INK2, length=0)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        w = 0.8 / len(series)
        for j, (name, ev) in enumerate(series):
            po = ev["summary"].get("policy_objects")
            if po and po != "all" and "Learned" in name:
                name = name.replace("Learned picks", f"Learned {po.replace(',', '+')} picks")
            rates = [ev["summary"]["subgoal_rates"].get(g, 0) * 100 for g in goals]
            full = ev["summary"]["success_rate"] * 100
            xs = np.arange(len(goals)) + (j - (len(series) - 1) / 2) * w
            ax.bar(xs, rates, width=w, color=cols[j], edgecolor=SURFACE, linewidth=2,
                   label=f"{name} — full task {full:.0f}%")
            for x, v in zip(xs, rates):
                ax.text(x, v + 1.5, f"{v:.0f}", ha="center", fontsize=7, color=INK2)
        ax.set_xticks(range(len(goals)), pretty, fontsize=8.5, color=INK)
        ax.set_ylim(0, 112)
        ax.set_yticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100%"], fontsize=8)
        ax.set_title("Sub-goal success over 10 randomized seeds", loc="left", color=INK, fontsize=11, pad=10)
        leg = ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
        for t in leg.get_texts():
            t.set_color(INK)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "subgoal_success.png"), facecolor=SURFACE, bbox_inches="tight", pad_inches=0.2)
        plt.close(fig)
        print("wrote subgoal_success.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/images")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--charts-only", action="store_true")
    ap.add_argument("--bench", default="results/benchmark.json")
    ap.add_argument("--eval-expert", default="results/eval_expert.json")
    ap.add_argument("--eval-heldout", default="results/eval_expert_seeds10-19.json")
    ap.add_argument("--eval-policy", default="results/eval_policy_int8.json")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    charts(a.out, a.bench, [("Scripted skills, seeds 0–9", a.eval_expert),
                            ("Scripted skills, held-out 10–19", a.eval_heldout),
                            ("Learned picks (INT8), seeds 0–9", a.eval_policy)])
    if not a.charts_only:
        randomization_grid(a.out)
        scene_shots(a.out, a.seed)


if __name__ == "__main__":
    main()
