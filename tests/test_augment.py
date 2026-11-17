"""Sensor augmentation: identity for an empty profile, bounded, and driven by the profile's own numbers."""
import pytest
import torch

from policy.augment import jitter_pixels, sensor_augment
from sim.config import realism_profile


def _batch(b=64):
    g = torch.Generator().manual_seed(0)
    return torch.rand(b, 2, 3, 24, 32, generator=g), torch.zeros(b, 12), torch.full((12,), 0.5)


def test_empty_profile_is_identity():
    imgs, st, sd = _batch()
    a, b = sensor_augment(imgs, st, {}, sd)
    assert torch.equal(a, imgs) and torch.equal(b, st)


def test_exposure_only_scales_by_a_gain_in_range():
    imgs = torch.full((32, 2, 3, 8, 8), 0.5)
    out, _ = sensor_augment(imgs, torch.zeros(32, 12), {"camera_exposure": [0.9, 1.1]}, torch.ones(12))
    gain = out[:, :, 0, 0, 0] / 0.5
    assert 0.9 - 1e-5 <= gain.min() and gain.max() <= 1.1 + 1e-5 and gain.std() > 0.01


def test_noise_has_the_profile_std_and_output_is_clipped():
    imgs = torch.full((64, 2, 3, 16, 16), 0.5)
    out, _ = sensor_augment(imgs, torch.zeros(64, 12), {"camera_noise_std": 2.0}, torch.ones(12))
    assert out.std().item() == pytest.approx(2.0 / 255, rel=0.1)
    out, _ = sensor_augment(torch.ones(4, 2, 3, 8, 8), torch.zeros(4, 12), {"camera_noise_std": 20.0}, torch.ones(12))
    assert out.max() <= 1.0 and out.min() >= 0.0


def test_joint_noise_is_in_standardised_units():
    _, st, sd = _batch(512)
    _, out = sensor_augment(torch.zeros(512, 2, 3, 8, 8), st, {"joint_noise_std": 0.002}, sd)
    assert out.std().item() == pytest.approx(0.002 / 0.5, rel=0.1)


def test_jitter_moves_the_image_by_at_most_the_profile_allows():
    imgs = torch.zeros(32, 2, 3, 24, 32)
    imgs[..., 10:14, 14:18] = 1.0                      # a bright patch
    prof = {"camera_jitter_deg": 2.0, "camera_jitter_mm": 5.0}
    out, _ = sensor_augment(imgs, torch.zeros(32, 12), prof, torch.ones(12))
    assert not torch.equal(out, imgs)
    assert jitter_pixels(prof, 24) == pytest.approx((2.0 + 0.477) * 24 / 50, rel=0.01)
    assert out.sum(dim=(2, 3, 4)).min() > 0            # the patch stays in view


def test_nominal_profile_runs_and_keeps_shapes():
    imgs, st, sd = _batch(8)
    a, b = sensor_augment(imgs, st, realism_profile("nominal"), sd)
    assert a.shape == imgs.shape and b.shape == st.shape and a.min() >= 0 and a.max() <= 1
