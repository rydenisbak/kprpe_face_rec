"""ViT-B with KP-RPE (keypoint relative position encoding), inference only.

Condensed from CVLface (MIT): models/vit_kprpe/{vit.py, rpe_options.py, RPE/KPRPE/*}, for the one configuration of
the released ViT-B WebFace12M checkpoint: ctx_type 'rel_keypoint_splithead_unshared', product method, ratio 1.9,
RPE on keys, contextual mode, no class token. Module names match the checkpoint keys, and the forward pass runs the
same tensor operations in the same order as the original. Training-only parts (drop path, dropout, token masking,
BatchNorm variant, weight init) and the other RPE variants are dropped.

Upstream of the CVLface code: the ViT backbone is InsightFace arcface_torch backbones/vit.py (MIT), the relative
position buckets are iRPE irpe.py from Microsoft Cream (MIT). Notices: THIRD_PARTY_NOTICES.md.
"""
import math

import torch
import torch.nn as nn

# released checkpoint: 112x112 input, 8x8 patches (14x14 tokens), width 512, 24 blocks, 16 heads, MLP x3, 512-d out
IMG_SIZE, PATCH, DIM, DEPTH, HEADS, MLP_RATIO, OUT_DIM = 112, 8, 512, 24, 16, 3, 512
NUM_KP = 5
RPE_RATIO = 1.9


# ---------------------------------------------------------------------------------------------------------------------
# relative position buckets (CVLface RPE/KPRPE/dist.py + kprpe_shared.py, product method)
# ---------------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def _piecewise_index(relative_position, alpha, beta, gamma, dtype):
    """Exact for |offset| <= alpha, log-compressed beyond, clipped to [-beta, beta]."""
    rp_abs = relative_position.abs()
    mask = rp_abs <= alpha
    not_mask = ~mask
    rp_out = relative_position[not_mask]
    rp_abs_out = rp_abs[not_mask]
    y_out = (torch.sign(rp_out) * (alpha + torch.log(rp_abs_out / alpha) / math.log(gamma / alpha) *
                                   (beta - alpha)).round().clip(max=beta)).to(dtype)
    idx = relative_position.clone()
    if idx.dtype in [torch.float32, torch.float64]:
        idx = idx.round().to(dtype)
    idx[not_mask] = y_out
    return idx


@torch.no_grad()
def rpe_gather_index(side, ratio=RPE_RATIO):
    """(L*L,) index into a per-query table (L, num_buckets) flattened: entry [i, j] -> i * num_buckets + bucket(i, j)."""
    alpha, beta, gamma = 1 * ratio, 2 * ratio, 8 * ratio
    beta_int = int(beta)
    s = 2 * beta_int + 1
    rows = torch.arange(side, dtype=torch.long).view(side, 1).repeat(1, side)
    cols = torch.arange(side, dtype=torch.long).view(1, side).repeat(side, 1)
    pos = torch.stack([rows, cols], 2).view(side * side, 1, 2)
    diff = pos - pos.view(1, side * side, 2)  # (L, L, 2) offsets (row, col) of query i relative to key j
    r = _piecewise_index(diff[:, :, 0], alpha, beta, gamma, torch.long) + beta_int
    c = _piecewise_index(diff[:, :, 1], alpha, beta, gamma, torch.long) + beta_int
    bucket = r * s + c  # (L, L) in [0, s*s)
    n = side * side
    offset = torch.arange(0, n * s * s, s * s, dtype=torch.long).view(-1, 1)
    return (bucket + offset).flatten()


@torch.no_grad()
def make_rel_keypoints(keypoints, query):
    """(B, 5, 2) keypoints (x, y) in [0, 1] -> (B, N, 10): offsets from every patch centre to every keypoint."""
    seq_length = query.shape[1]
    side = int(math.sqrt(seq_length))
    assert side == math.sqrt(seq_length)
    coord = torch.linspace(0, 1, side + 1, device=query.device, dtype=query.dtype)
    coord = (coord[:-1] + coord[1:]) / 2  # patch centres
    x, y = torch.meshgrid(coord, coord, indexing='ij')
    grid = torch.stack([y, x], dim=-1).reshape(-1, 2).unsqueeze(0).unsqueeze(-2)  # (1, N, 1, 2), (x, y) order
    diff = grid - keypoints.unsqueeze(-3)  # (B, N, 5, 2)
    return diff.flatten(2)


# ---------------------------------------------------------------------------------------------------------------------
# transformer
# ---------------------------------------------------------------------------------------------------------------------
class Mlp(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.ReLU6()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, rpe_table, rpe_index):
        """rpe_table: (B, heads, N, num_buckets) keypoint-conditioned bias table of this block."""
        batch_size, num_token, embed_dim = x.shape
        qkv = self.qkv(x).reshape(batch_size, num_token, 3, self.num_heads,
                                  embed_dim // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q *= self.scale
        attn = (q @ k.transpose(-2, -1))
        attn += rpe_table.flatten(2)[:, :, rpe_index].view(len(rpe_table), -1, num_token, num_token)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(batch_size, num_token, embed_dim)
        return self.proj(x)


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x, rpe_table, rpe_index):
        x = x + self.attn(self.norm1(x), rpe_table, rpe_index)
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(3, DIM, kernel_size=PATCH, stride=PATCH)

    def forward(self, x):
        assert x.shape[2:] == (IMG_SIZE, IMG_SIZE), f'input must be {IMG_SIZE}x{IMG_SIZE}, got {tuple(x.shape[2:])}'
        return self.proj(x).flatten(2).transpose(1, 2)


class ViTKPRPE(nn.Module):
    """forward(x, keypoints): x (B, 3, 112, 112) RGB in [-1, 1]; keypoints (B, 5, 2) (x, y) in [0, 1] of the crop
    -> (B, 512) raw (not normalized) embedding."""

    def __init__(self):
        super().__init__()
        side = IMG_SIZE // PATCH
        self.num_patches = side * side
        self.num_buckets = (2 * int(2 * RPE_RATIO) + 1) ** 2  # 7x7 relative offset cells
        self.patch_embed = PatchEmbed()
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, DIM))
        self.blocks = nn.ModuleList([Block(DIM, HEADS, MLP_RATIO) for _ in range(DEPTH)])
        self.norm = nn.LayerNorm(DIM)
        self.feature = nn.Sequential(
            nn.Linear(DIM * self.num_patches, DIM, bias=False),
            nn.BatchNorm1d(DIM, eps=2e-5),
            nn.Linear(DIM, OUT_DIM, bias=False),
            nn.BatchNorm1d(OUT_DIM, eps=2e-5))
        # patch -> keypoint offsets (10) -> a bias table (num_buckets) for every block and head
        self.keypoint_linear = nn.Linear(2 * NUM_KP, self.num_buckets * HEADS * DEPTH)
        self.register_buffer('rpe_index', rpe_gather_index(side), persistent=False)

    def forward(self, x, keypoints):
        batch_size = x.shape[0]
        x = self.patch_embed(x)
        x = x + self.pos_embed
        rel = make_rel_keypoints(keypoints, x)[:, :, :2 * NUM_KP]
        rel = self.keypoint_linear(rel)
        rel = rel.view(batch_size, -1, HEADS * DEPTH, self.num_buckets).transpose(1, 2)
        for block, table in zip(self.blocks, torch.chunk(rel, DEPTH, dim=1)):
            x = block(x, table, self.rpe_index)
        x = self.norm(x.float())
        x = torch.reshape(x, (batch_size, self.num_patches * DIM))
        return self.feature(x)
