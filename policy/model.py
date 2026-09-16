"""ACT-Lite: a compact language-conditioned Action-Chunking Transformer for the dual SO-101 arms.

Inputs  : two RGB cameras (overhead, front), 12-D joint state, tokenised sub-instruction.
Outputs : a chunk of K future 12-D joint targets (both arms, incl. grippers), normalised.

Design notes
  * ACT (Zhao et al. 2023) without the CVAE branch (deterministic L1 regression) - trains in minutes
    on a single GPU from a few hundred scripted demos and exports cleanly to OpenVINO IR.
  * Small ResNet-style CNN backbone (no pretrained weights needed, ~1.9M params total) so the whole
    policy runs in a few ms on an Intel Core Ultra CPU/iGPU/NPU after INT8 quantisation.
  * Language is a word-level vocabulary built from the planner's canonical sub-instructions;
    the LLM/rule planner handles free-form text, the policy consumes the canonical form.
"""
from __future__ import annotations

import re

import torch
import torch.nn as nn

VOCAB = ["<pad>", "<unk>", "arm", "a", "b", "pick", "up", "the", "place", "on", "its", "spot", "open", "drawer",
         "hand", "to", "pour", "from", "bottle", "into", "mug", "held", "by", "plate", "fork", "spoon"]
W2I = {w: i for i, w in enumerate(VOCAB)}
MAX_TOK = 14


def tokenize(text: str) -> list[int]:
    ids = [W2I.get(w, 1) for w in re.findall(r"[a-z]+", text.lower())][:MAX_TOK]
    return ids + [0] * (MAX_TOK - len(ids))


class ConvBackbone(nn.Module):
    def __init__(self, dim=128):
        super().__init__()

        def block(ci, co, s):
            return nn.Sequential(nn.Conv2d(ci, co, 3, s, 1, bias=False), nn.GroupNorm(8, co), nn.ReLU(inplace=True),
                                 nn.Conv2d(co, co, 3, 1, 1, bias=False), nn.GroupNorm(8, co), nn.ReLU(inplace=True))
        self.net = nn.Sequential(block(3, 32, 2), block(32, 64, 2), block(64, 96, 2), block(96, dim, 2))

    def forward(self, x):          # (B,3,96,128) -> (B,dim,6,8)
        return self.net(x)


class ACTLite(nn.Module):
    def __init__(self, n_cams=2, state_dim=12, act_dim=12, chunk=20, dim=128, heads=4, enc_layers=3, dec_layers=2,
                 img_hw=(96, 128)):
        super().__init__()
        self.chunk, self.act_dim, self.n_cams = chunk, act_dim, n_cams
        self.backbone = ConvBackbone(dim)
        fh, fw = img_hw[0] // 16, img_hw[1] // 16
        self.img_pos = nn.Parameter(torch.randn(1, n_cams * fh * fw, dim) * 0.02)
        self.state_proj = nn.Linear(state_dim, dim)
        self.tok_emb = nn.Embedding(len(VOCAB), dim, padding_idx=0)
        self.txt_pos = nn.Parameter(torch.randn(1, MAX_TOK, dim) * 0.02)
        self.type_emb = nn.Parameter(torch.randn(1, 3, dim) * 0.02)   # image / state / text
        enc = nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout=0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, enc_layers, enable_nested_tensor=False)
        dec = nn.TransformerDecoderLayer(dim, heads, dim * 4, dropout=0.1, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(dec, dec_layers)
        self.queries = nn.Parameter(torch.randn(1, chunk, dim) * 0.02)
        self.head = nn.Linear(dim, act_dim)
        self.register_buffer("img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, images, state, tokens):
        """images: (B, n_cams, 3, H, W) float in [0,1]; state: (B,12) normalised; tokens: (B, MAX_TOK) int64."""
        B = state.shape[0]
        x = images.flatten(0, 1)
        x = (x - self.img_mean) / self.img_std
        f = self.backbone(x)                                   # (B*n, D, h, w)
        f = f.flatten(2).transpose(1, 2)                       # (B*n, h*w, D)
        f = f.reshape(B, -1, f.shape[-1]) + self.img_pos + self.type_emb[:, 0:1]
        s = self.state_proj(state).unsqueeze(1) + self.type_emb[:, 1:2]
        t = self.tok_emb(tokens) + self.txt_pos + self.type_emb[:, 2:3]
        mem = self.encoder(torch.cat([f, s, t], 1))
        q = self.queries.expand(B, -1, -1)
        h = self.decoder(q, mem)
        return self.head(h)                                    # (B, chunk, act_dim)


def count_params(m):
    return sum(p.numel() for p in m.parameters())
