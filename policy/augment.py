"""Training-time augmentation that mirrors the sensor effects of a realism profile (sim/config.yaml).

The simulator's `nominal` profile perturbs what the policy sees: per-pixel camera noise, occasional blur, an exposure
gain, a small camera pose offset, and joint-reading noise. `sensor_augment` applies the same kinds of perturbation
to a training batch from the profile's own numbers, so the augmentation strength follows the deployment-time noise
instead of being a separate guess. Camera pose offsets are approximated as an image-plane rotation and translation.
Everything runs on the batch tensors (GPU if the batch is there).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

FOVY_DEG = 50.0          # the policy cameras are 50-55 degrees vertical (sim/scene.py)
CAMERA_DISTANCE = 0.6    # m, rough camera-to-workspace distance, to turn a position offset into an angle


def jitter_pixels(profile: dict, height: int) -> float:
    """Largest image shift in pixels that the profile's camera pose jitter can cause."""
    deg = float(profile.get("camera_jitter_deg", 0.0))
    deg += math.degrees(float(profile.get("camera_jitter_mm", 0.0)) / 1000.0 / CAMERA_DISTANCE)
    return deg * height / FOVY_DEG


def sensor_augment(imgs: torch.Tensor, state: torch.Tensor, profile: dict, s_std: torch.Tensor,
                   gen: torch.Generator | None = None):
    """imgs (B, V, 3, H, W) in [0, 1]; state (B, D) standardised by `s_std` (raw units divided by s_std).

    Returns augmented copies. An empty profile returns the inputs unchanged."""
    if not profile:
        return imgs, state
    B, V, C, H, W = imgs.shape
    dev = imgs.device

    def u(lo, hi, *shape):
        return lo + (hi - lo) * torch.rand(*shape, device=dev, generator=gen)

    x = imgs.reshape(B * V, C, H, W)
    if profile.get("camera_exposure"):
        lo, hi = profile["camera_exposure"]
        x = x * u(lo, hi, B * V, 1, 1, 1)
    deg, px = float(profile.get("camera_jitter_deg", 0.0)), jitter_pixels(profile, H)
    if deg or px:
        ang = u(-deg, deg, B * V) * math.pi / 180.0
        tx, ty = u(-px, px, B * V) * 2.0 / W, u(-px, px, B * V) * 2.0 / H     # normalised coordinates span [-1, 1]
        c, s = torch.cos(ang), torch.sin(ang)
        theta = torch.stack([torch.stack([c, -s, tx], 1), torch.stack([s, c, ty], 1)], 1)
        grid = F.affine_grid(theta, list(x.shape), align_corners=False)
        x = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
    p_blur = float(profile.get("camera_blur_prob", 0.0))
    if p_blur:
        blurred = F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), 3, stride=1)
        use = (torch.rand(B * V, 1, 1, 1, device=dev, generator=gen) < p_blur).to(x.dtype)
        x = use * blurred + (1 - use) * x
    if profile.get("camera_noise_std"):
        x = x + torch.randn(x.shape, device=dev, generator=gen) * (float(profile["camera_noise_std"]) / 255.0)
    imgs = x.clamp(0, 1).reshape(B, V, C, H, W)
    if profile.get("joint_noise_std"):
        state = state + torch.randn(state.shape, device=dev, generator=gen) * float(profile["joint_noise_std"]) / s_std
    return imgs, state
