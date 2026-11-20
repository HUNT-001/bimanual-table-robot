"""Object-pose CNN: one convolutional trunk per camera with a stride-4 feature map, spatial-softmax keypoints
(sub-pixel image positions, the first few supervised with the projected object positions), and an MLP from the
keypoints of both cameras plus pooled features to the label vector of perception/labels.py."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn(cin, cout, k=3, s=1):
    return nn.Sequential(nn.Conv2d(cin, cout, k, s, k // 2, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class Res(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.a, self.b = conv_bn(c, c), nn.Sequential(nn.Conv2d(c, c, 3, 1, 1, bias=False), nn.BatchNorm2d(c))

    def forward(self, x):
        return F.relu(x + self.b(self.a(x)))


class Trunk(nn.Module):
    """240x320 -> features at 1/4 (FPN-merged), and a pooled vector from 1/32."""

    def __init__(self, width=32, fpn=96, k=16):
        super().__init__()
        w = width
        self.stem = nn.Sequential(conv_bn(3, w, 5, 2), Res(w))                      # 1/2
        self.s4 = nn.Sequential(conv_bn(w, 2 * w, 3, 2), Res(2 * w), Res(2 * w))     # 1/4
        self.s8 = nn.Sequential(conv_bn(2 * w, 4 * w, 3, 2), Res(4 * w), Res(4 * w))  # 1/8
        self.s16 = nn.Sequential(conv_bn(4 * w, 8 * w, 3, 2), Res(8 * w), Res(8 * w))  # 1/16
        self.s32 = nn.Sequential(conv_bn(8 * w, 8 * w, 3, 2), Res(8 * w))            # 1/32
        self.l4, self.l8, self.l16 = (nn.Conv2d(c, fpn, 1) for c in (2 * w, 4 * w, 8 * w))
        self.merge = nn.Sequential(conv_bn(fpn, fpn), nn.Conv2d(fpn, k, 1))
        self.out_dim = 8 * w

    def forward(self, x):
        x = self.stem(x)
        f4 = self.s4(x); f8 = self.s8(f4); f16 = self.s16(f8); f32 = self.s32(f16)
        p = self.l16(f16)
        p = self.l8(f8) + F.interpolate(p, size=f8.shape[-2:], mode="bilinear", align_corners=False)
        p = self.l4(f4) + F.interpolate(p, size=f4.shape[-2:], mode="bilinear", align_corners=False)
        return self.merge(p), f32.mean((2, 3))


def spatial_softmax(logits, temp):
    """Expected normalised (x, y) in [-1, 1] of each channel (pixel-centre convention), in float32."""
    b, k, h, w = logits.shape
    lg = logits.float().flatten(2) / temp.view(1, k, 1)
    p = lg.softmax(-1).view(b, k, h, w)
    xs = (torch.arange(w, device=logits.device, dtype=torch.float32) + 0.5) / w * 2 - 1
    ys = (torch.arange(h, device=logits.device, dtype=torch.float32) + 0.5) / h * 2 - 1
    ex = (p.sum(2) * xs).sum(-1)
    ey = (p.sum(3) * ys).sum(-1)
    peak = lg.max(-1).values - lg.logsumexp(-1)          # log of the peak probability: a confidence
    return torch.stack([ex, ey], -1), peak


class PoseNet(nn.Module):
    def __init__(self, n_out, n_cams=2, k=16, width=32, label_mean=None, label_std=None):
        super().__init__()
        self.n_cams, self.k = n_cams, k
        self.trunks = nn.ModuleList(Trunk(width, k=k) for _ in range(n_cams))
        self.log_temp = nn.Parameter(torch.zeros(n_cams, k))
        d_in = n_cams * (k * 3 + self.trunks[0].out_dim)
        self.head = nn.Sequential(nn.Linear(d_in, 512), nn.ReLU(inplace=True), nn.Linear(512, 512),
                                  nn.ReLU(inplace=True), nn.Linear(512, n_out))
        self.lin = nn.Linear(n_cams * k * 2, n_out)      # near-affine image -> table map, learned directly
        self.register_buffer("mu", torch.zeros(n_out) if label_mean is None else torch.as_tensor(label_mean).float())
        self.register_buffer("sd", torch.ones(n_out) if label_std is None else torch.as_tensor(label_std).float())

    def forward(self, imgs):
        """imgs: (B, cams, H, W, 3), uint8 or float in [0, 1]. Returns (labels (B, n_out) in label units,
        keypoints (B, cams, k, 2)). Permuting HWC to NCHW gives channels_last strides without a copy."""
        kps, feats = [], []
        for c in range(self.n_cams):
            x = imgs[:, c].permute(0, 3, 1, 2)
            x = (x.float() / 255.0 if x.dtype == torch.uint8 else x.float()) - 0.5
            logits, g = self.trunks[c](x.contiguous(memory_format=torch.channels_last))
            kp, peak = spatial_softmax(logits, self.log_temp[c].exp().clamp(min=0.05))
            kps.append(kp)
            feats += [kp.flatten(1), peak, g.float()]
        kp = torch.stack(kps, 1)
        with torch.autocast(device_type=imgs.device.type, enabled=False):
            z = self.head(torch.cat(feats, 1).float()) + self.lin(kp.flatten(1))
        return self.mu + self.sd * z, kp
