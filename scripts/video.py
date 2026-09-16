"""Annotated demo-video writer: command, current sub-step, seed/randomization, live checklist."""
import os

import imageio.v2 as imageio
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover
    Image = None


def _font(size):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "C:/Windows/Fonts/arial.ttf",
              "/System/Library/Fonts/Supplemental/Arial.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


class VideoWriter:
    def __init__(self, path, instruction, seed, every=2, hw=(480, 640), fps=20, cams=("demo", "overhead")):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.path, self.instruction, self.seed, self.every = path, instruction, seed, every
        self.hw, self.cams = hw, cams
        self.w = imageio.get_writer(path, fps=max(1, fps // every), codec="libx264", quality=7, macro_block_size=1)
        self.f_big, self.f_small = _font(18), _font(13)
        self.n = 0

    def _compose(self, env, step_text, banner=None):
        main = env.render_frame(self.cams[0], self.hw)
        small = env.render_frame(self.cams[1], (self.hw[0] // 2, self.hw[1] // 2)) if len(self.cams) > 1 else None
        H, W = self.hw
        canvas = Image.new("RGB", (W + W // 2, H + 70), (18, 18, 24))
        canvas.paste(Image.fromarray(main), (0, 70))
        if small is not None:
            canvas.paste(Image.fromarray(small), (W, 70))
        d = ImageDraw.Draw(canvas)
        # command (wrapped)
        words, lines, cur = self.instruction.split(), [], ""
        for w in words:
            if d.textlength(cur + " " + w, font=self.f_small) > (W + W // 2) - 16:
                lines.append(cur); cur = w
            else:
                cur = (cur + " " + w).strip()
        lines.append(cur)
        for i, l in enumerate(lines[:3]):
            d.text((8, 4 + 16 * i), l, font=self.f_small, fill=(230, 230, 230))
        d.text((8, 52), f"seed {self.seed} | t={env.t / 20:5.1f}s | now: {step_text}", font=self.f_small,
               fill=(255, 210, 90))
        # checklist panel
        p = env.info.params
        y = 70 + H // 2 + 8
        checks = [("drawer open", env.check("drawer_open"))]
        for o in ("plate", "spoon", "fork", "mug", "bottle"):
            checks.append((f"{o} on spot", env.check("on_target", o)))
        checks.append((f"poured ({env.poured:.1f}s)", env.check("poured")))
        for name, ok in checks:
            d.text((W + 12, y), ("[x] " if ok else "[ ] ") + name, font=self.f_small,
                   fill=(120, 230, 120) if ok else (200, 200, 200))
            y += 17
        y += 6
        info = [f"held A: {env.held['A']}", f"held B: {env.held['B']}",
                f"light {p['light_intensity']:.2f}  table mu {p['table_friction']:.2f}",
                f"mug {'box' if p['mug_shape'] else 'cyl'} {p['mug_mass'] * 1000:.0f} g  bottle {p['bottle_mass'] * 1000:.0f} g",
                f"size x{p['size_scale']:.2f}"]
        for l in info:
            d.text((W + 12, y), l, font=self.f_small, fill=(160, 180, 220)); y += 16
        if banner:
            d.rectangle([0, 70 + H // 2 - 24, W, 70 + H // 2 + 24], fill=(0, 0, 0))
            d.text((20, 70 + H // 2 - 12), banner, font=self.f_big, fill=(255, 255, 255))
        return np.asarray(canvas)

    def on_step(self, env, step_text):
        self.n += 1
        if self.n % self.every == 0:
            self.w.append_data(self._compose(env, step_text))

    def finish(self, env, res):
        ok = sum(res.subgoals.values())
        banner = f"{'SUCCESS' if res.success else 'PARTIAL'}: {ok}/{len(res.subgoals)} goals  (seed {self.seed})"
        frame = self._compose(env, "done", banner)
        for _ in range(15):
            self.w.append_data(frame)
        self.w.close()
