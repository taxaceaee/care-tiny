"""CARE-Tiny detector: a dense periphery at stride >= 2 routes a few candidates to window readouts that place every
box edge on the pixel grid.

Pipeline (one forward pass):
    image -> HGNetv2 backbone -> FPN + PAN neck -> decoupled dense head on a stride-2 sub-slot grid
          -> candidate routing (M candidates, K refined per image, K from the estimated object count)
          -> two readouts on 16 x 16-cell windows sampled from shared maps (stride-2 detail F2, stride-1 pixels,
             PAN semantics); the second window is re-centred on the first readout's box and its queries first
             attend to their k nearest neighbours -> refined boxes + the unrefined candidates -> NMS.

Shapes: B images, Hs x Ws stride-2 sub-slot grid (Ns = Hs * Ws), Q queries (K refined + DN), W window cells,
fd window feature width, D query width. Coordinates are input pixels; boxes are xyxy unless named cxcywh.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import batched_nms

from . import box_ops
from .backbone import Backbone

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm over the channels of an NCHW (or channels_last) map.

    keep_dtype: return the input dtype. Under autocast layer_norm computes and returns fp32; maps that are only
    sampled or fed to the next conv / einsum are rounded back to the autocast dtype instead of being kept in fp32.
    """

    def __init__(self, *args, keep_dtype=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.keep_dtype = keep_dtype

    def forward(self, x):
        y = F.layer_norm(x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps)
        if self.keep_dtype:
            y = y.to(x.dtype)
        return y.permute(0, 3, 1, 2)


# ----------------------------------------------------------------------------- periphery
class ConvNeXtBlock(nn.Module):
    """DW kxk -> LN -> PW (expand) -> GELU -> PW, residual."""

    def __init__(self, dim, k=7, expand=2):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, k, padding=k // 2, groups=dim)
        self.norm = LayerNorm2d(dim)
        self.pw1 = nn.Conv2d(dim, expand * dim, 1)
        self.pw2 = nn.Conv2d(expand * dim, dim, 1)

    def forward(self, x):
        return x + self.pw2(F.gelu(self.pw1(self.norm(self.dw(x)))))


class SceneContext(nn.Module):
    """64 global scene tokens Z from the deepest backbone level, shared by the neck FiLM and the readouts."""

    def __init__(self, c_top, qdim, heads, layers=1):
        super().__init__()
        self.z_proj = nn.Linear(c_top, qdim)
        self.z_pos = nn.Parameter(torch.zeros(1, 64, qdim))
        nn.init.trunc_normal_(self.z_pos, std=0.02)
        layer = nn.TransformerEncoderLayer(qdim, heads, 2 * qdim, dropout=0.0, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.z_enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.z_norm = nn.LayerNorm(qdim)

    def forward(self, f_top):
        z = F.adaptive_avg_pool2d(f_top, 8).flatten(2).transpose(1, 2)
        return self.z_norm(self.z_enc(self.z_proj(z) + self.z_pos))


class FPNNeck(nn.Module):
    """Top-down FPN to stride 4 with ConvNeXt fusion blocks at every level, then a bottom-up PAN path, so every
    level carries both the stride-4 detail and the deep semantics. The stride-4 level also receives the raw 4x4
    pixels of each cell and a FiLM from the scene tokens.

    Returns the stride-4 map, the scene tokens Z and the list of PAN levels (stride 4 .. 32)."""

    def __init__(self, in_channels, dim, qdim, heads, blocks=(2, 1, 1, 1), scene_layers=1):
        """blocks: fusion blocks at stride 4, 8, 16(, 32), one entry per backbone level."""
        super().__init__()
        n_lvl = len(in_channels)
        blocks = tuple(blocks)[:n_lvl]
        assert len(blocks) == n_lvl, f"neck_blocks needs {n_lvl} entries, got {blocks}"
        self.top = n_lvl - 1
        self.lateral = nn.ModuleList([nn.Sequential(nn.Conv2d(c, dim, 1), LayerNorm2d(dim)) for c in in_channels])
        self.fuse = nn.ModuleList([nn.Sequential(*[ConvNeXtBlock(dim) for _ in range(n)]) for n in blocks[::-1]])
        self.raw = nn.Sequential(nn.Conv2d(48, dim, 1), LayerNorm2d(dim))
        self.ctx = SceneContext(in_channels[-1], qdim, heads, scene_layers)
        self.film = nn.Linear(qdim, 2 * dim)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.down = nn.ModuleList([nn.Sequential(nn.Conv2d(dim, dim, 3, stride=2, padding=1, groups=dim),
                                                 LayerNorm2d(dim), nn.Conv2d(dim, dim, 1))
                                   for _ in range(self.top)])
        self.up_fuse = nn.ModuleList([ConvNeXtBlock(dim) for _ in range(self.top)])

    def forward(self, feats, raw):
        top = self.top
        z = self.ctx(feats[top])
        p = self.fuse[0](self.lateral[top](feats[top]))  # fuse[i] runs at level top - i (deepest first)
        td = [p]
        for i, lvl in enumerate(range(top - 1, -1, -1), start=1):
            lat = self.lateral[lvl](feats[lvl])
            p = lat + F.interpolate(p, size=lat.shape[-2:], mode="bilinear", align_corners=False)
            if lvl == 0:
                p = p + self.raw(raw)
                g, b = self.film(z.mean(1)).chunk(2, -1)
                p = p * (1 + g[..., None, None]) + b[..., None, None]
            p = self.fuse[i](p)
            td.append(p)
        td = td[::-1]  # stride 4 first
        outs = [p]
        for lvl in range(1, top + 1):
            d = self.down[lvl - 1](outs[-1])
            outs.append(self.up_fuse[lvl - 1](td[lvl] + d[..., : td[lvl].shape[-2], : td[lvl].shape[-1]]))
        return p, z, outs


class DecoupledDenseHead(nn.Module):
    """Separate objectness / class and box towers (n x [DW 3x3 -> LN -> PW -> GELU]) at stride 4, 4 sub-slots per
    cell pixel-shuffled to the stride-2 grid. Output channel = field * 4 + sub-slot, fields obj | cls | box."""

    def __init__(self, dim, num_classes, convs=2):
        super().__init__()
        self.C = num_classes

        def tower():
            layers = []
            for _ in range(convs):
                layers += [nn.Conv2d(dim, dim, 3, padding=1, groups=dim), LayerNorm2d(dim), nn.Conv2d(dim, dim, 1),
                           nn.GELU()]
            return nn.Sequential(*layers)

        self.cls_tower, self.box_tower = tower(), tower()
        self.cls_out = nn.Conv2d(dim, 4 * (1 + num_classes), 1)
        self.box_out = nn.Conv2d(dim, 4 * 4, 1)
        nn.init.normal_(self.cls_out.weight, std=0.01)
        nn.init.constant_(self.cls_out.bias, -4.6)  # prior probability 0.01 for objectness and classes
        nn.init.normal_(self.box_out.weight, std=0.01)
        nn.init.zeros_(self.box_out.bias)

    def forward(self, h):
        y = torch.cat([self.cls_out(self.cls_tower(h)), self.box_out(self.box_tower(h))], 1)
        y = F.pixel_shuffle(y, 2)
        return y[:, :1], y[:, 1:1 + self.C], y[:, 1 + self.C:]


class HiResMap(nn.Module):
    """Shared stride-2 detail map F2, computed once per image from the stride-2 stem activation, the 2x2 raw pixels
    of each cell and the upsampled stride-4 neck map. The query windows only sample it."""

    def __init__(self, s2_ch, neck_dim, dim=32, blocks=1, k=7):
        super().__init__()
        self.in_proj = nn.Conv2d(s2_ch + 12, dim, 1)
        self.in_norm = LayerNorm2d(dim)
        self.p4_proj = nn.Conv2d(neck_dim, dim, 1)
        self.blocks = nn.Sequential(*[ConvNeXtBlock(dim, k=k) for _ in range(blocks)])
        self.out_norm = LayerNorm2d(dim, keep_dtype=True)

    def forward(self, s2, x, p4):
        """s2 (B,c,H/2,W/2) stem activation, x (B,3,H,W) normalised image, p4 (B,D,H/4,W/4) -> (B,dim,H/2,W/2)."""
        a = self.in_norm(self.in_proj(torch.cat([s2.to(x.dtype), F.pixel_unshuffle(x, 2)], 1)))
        a = a + F.interpolate(self.p4_proj(p4), size=a.shape[-2:], mode="bilinear", align_corners=False)
        return self.out_norm(self.blocks(a))


class PixelMap(nn.Module):
    """Shared stride-1 map (3x3 conv on the normalised image + GELU + 1x1 + LN): the stride-2 F2 cannot place an
    edge of a 2-8 px object to the pixel. About 1 GFLOP at 800 x 800; the windows only sample it."""

    def __init__(self, dim=8):
        super().__init__()
        self.conv = nn.Conv2d(3, dim, 3, padding=1)
        self.pw = nn.Conv2d(dim, dim, 1)
        self.norm = LayerNorm2d(dim, keep_dtype=True)

    def forward(self, x):
        return self.norm(self.pw(F.gelu(self.conv(x))))


def decode_dense(box_raw):
    """(B,4,Hs,Ws) -> (B,Ns,4) cxcywh. Sub-cell (x, y) covers [2x, 2x + 2); its centre may move to [2x - 1, 2x + 3)."""
    r = box_raw.float()
    B, _, Hs, Ws = r.shape
    ys = torch.arange(Hs, device=r.device, dtype=torch.float32).view(1, Hs, 1)
    xs = torch.arange(Ws, device=r.device, dtype=torch.float32).view(1, 1, Ws)
    cx = 2 * xs + 4 * torch.sigmoid(r[:, 0]) - 1
    cy = 2 * ys + 4 * torch.sigmoid(r[:, 1]) - 1
    w = 8 * torch.exp(r[:, 2].clamp(-4, 4))
    h = 8 * torch.exp(r[:, 3].clamp(-4, 4))
    return torch.stack([cx, cy, w, h], -1).view(B, Hs * Ws, 4)


@torch.no_grad()
def count_objects(obj_logit, Hs, Ws, thr):
    """Estimated object count per image: 3x3 local maxima of the objectness above thr."""
    p = torch.sigmoid(obj_logit).view(-1, 1, Hs, Ws)
    peak = (p == F.max_pool2d(p, 3, 1, 1)) & (p > thr)
    return peak.flatten(1).sum(1)


@torch.no_grad()
def select_candidates(score_logit, Hs, Ws, M, R, qpr, k_list, spatial_frac, kmax=None, region_min_prob=0.0):
    """Keep M sub-slots (a quota per region of an R x R grid + the global top), then pick K per image to refine.

    Inside the K pick, spatial_frac of K is reserved for region winners above region_min_prob, and 3x3 local maxima
    of the score rank above every non-peak: each object has ~10 positive sub-slots, so a plain top-K spends K on
    duplicates of confident objects while weaker objects get no refined query. Non-peaks fill the rest of K.

    kmax: max(k_list) when the caller knows it on the host (avoids a .item() sync).
    Returns candM (B,M) sub-slot ids, kidx (B,Kb) positions inside candM, active (B,Kb) mask.
    """
    B, N = score_logit.shape
    prob = torch.sigmoid(score_logit)
    rh, rw = math.ceil(Hs / R), math.ceil(Ws / R)
    pad = F.pad(prob.view(B, Hs, Ws), (0, R * rw - Ws, 0, R * rh - Hs), value=-1.0)
    reg = pad.view(B, R, rh, R, rw).permute(0, 1, 3, 2, 4).reshape(B, R * R, rh * rw)
    q = min(qpr, rh * rw)
    qv, qi = reg.topk(q, dim=-1)
    rid = torch.arange(R * R, device=prob.device)
    gy = (rid // R).view(1, -1, 1) * rh + qi // rw
    gx = (rid % R).view(1, -1, 1) * rw + qi % rw
    ok = (qv >= 0) & (gy < Hs) & (gx < Ws)
    gidx = (gy.clamp(max=Hs - 1) * Ws + gx.clamp(max=Ws - 1)).view(B, -1)
    boost = torch.zeros_like(prob).scatter_reduce(1, gidx, 2.0 * ok.view(B, -1).float(), "amax")
    top1 = torch.zeros_like(prob).scatter_reduce(1, gidx[:, ::q], ok[..., 0].float(), "amax")
    candM = (prob + boost).topk(M, dim=1).indices
    probM = prob.gather(1, candM)
    t1 = top1.gather(1, candM) > 0
    pr = torch.where(t1, probM, torch.full_like(probM, -1.0))
    order = pr.argsort(1, descending=True)
    rank = torch.empty_like(order).scatter_(1, order, torch.arange(M, device=prob.device).expand(B, M))
    kq = (spatial_frac * k_list.float()).long()
    pick = t1 & (rank < kq[:, None])
    if region_min_prob > 0:
        pick = pick & (probM > region_min_prob)
    if kmax is None:
        kmax = int(k_list.max().item())
    rank_key = probM + 2.0 * pick.float()
    p2 = prob.view(B, 1, Hs, Ws)
    peak_thr = max(float(region_min_prob), 0.05)  # background ripples are no peaks
    peak = ((p2 == F.max_pool2d(p2, 3, 1, 1)) & (p2 > peak_thr)).view(B, N).gather(1, candM)
    rank_key = rank_key + 1.0 * peak.float()
    kidx = rank_key.topk(kmax, dim=1).indices
    active = torch.arange(kmax, device=prob.device)[None, :] < k_list[:, None]
    return candM, kidx, active


# ----------------------------------------------------------------------------- sampling
def _sample(feat, xs, ys, stride):
    """Bilinear samples of feat (B,C,Hf,Wf) at image-pixel grids xs (B,Q,nx), ys (B,Q,ny) -> (B,Q,C,ny,nx)."""
    B, C, Hf, Wf = feat.shape
    Q, nx, ny = xs.shape[1], xs.shape[-1], ys.shape[-1]
    gx = xs / (stride * Wf) * 2 - 1
    gy = ys / (stride * Hf) * 2 - 1
    gx, gy = torch.broadcast_tensors(gx[:, :, None, :], gy[:, :, :, None])
    grid = torch.stack([gx, gy], -1).reshape(B, Q * ny, nx, 2)
    out = F.grid_sample(feat, grid.to(feat.dtype), mode="bilinear", padding_mode="zeros", align_corners=False)
    return out.view(B, C, Q, ny, nx).permute(0, 2, 1, 3, 4)


def build_pyramid(x, levels):
    pyr = [x]
    for _ in range(1, levels):
        pyr.append(F.avg_pool2d(pyr[-1], 2, ceil_mode=True))
    return pyr


def window_geometry(box, W, W_obj, levels):
    """box cxcywh (B,Q,4) -> window origin X0, Y0 (px), cell size sig (px) and pyramid level of every query.
    Windows of objects up to W_obj px use native pixels (sig = 1) and integer origins."""
    cx, cy, w, h = box.unbind(-1)
    sig = (torch.maximum(w, h) / W_obj).clamp(1.0, 2.0 ** (levels - 1))
    # nan_to_num: a non-finite box must give a non-finite loss (the step is then skipped), not a NaN -> long index
    lvl = torch.nan_to_num(torch.floor(torch.log2(sig) + 1e-6), nan=0.0).clamp(0, levels - 1).long()
    half = W / 2
    native = sig <= 1.0
    X0 = torch.where(native, torch.round(cx) - half, cx - half * sig)
    Y0 = torch.where(native, torch.round(cy) - half, cy - half * sig)
    return {"X0": X0, "Y0": Y0, "sig": sig, "lvl": lvl}


def semantic_level(box, obj_cells, n_levels):
    """Index (stride 4 * 2**index) of the PAN level where the object spans obj_cells..2*obj_cells feature pixels
    (FPN RoI level rule), box cxcywh (B,Q,4) -> (B,Q) long in [0, n_levels)."""
    size = torch.maximum(box[..., 2], box[..., 3]).clamp(min=1.0)
    li = torch.floor(torch.log2(size / (4.0 * obj_cells)) + 1e-6)
    return torch.nan_to_num(li, nan=0.0).clamp(0, n_levels - 1).long()


ATLAS_GAP = 2  # zero pixels between pyramid levels in the atlas (bilinear taps reach at most 1 px outside a level)
_ATLAS_TABLES = {}


def pyramid_atlas(pyr):
    """All pyramid levels in one zero canvas: level 0 at (0, 0), levels 1.. stacked in a column to its right with
    ATLAS_GAP zero rows / columns in between. Returns the canvas (B,C,Ha,Wa) and a (levels, 4) device table of
    (x offset, y offset, width, height) per level (cached per shape: no host -> device copy per step)."""
    B, C, H0, W0 = pyr[0].shape
    rows, y = [(0, 0, W0, H0)], 0
    for P in pyr[1:]:
        rows.append((W0 + ATLAS_GAP, y, P.shape[-1], P.shape[-2]))
        y += P.shape[-2] + ATLAS_GAP
    Ha = max(H0, y)
    Wa = W0 + (ATLAS_GAP + max(r[2] for r in rows[1:]) if len(rows) > 1 else 0)
    canvas = pyr[0].new_zeros(B, C, Ha, Wa)
    for P, (ox, oy, w, h) in zip(pyr, rows):
        canvas[:, :, oy:oy + h, ox:ox + w] = P
    key = (str(canvas.device), tuple(rows))
    if key not in _ATLAS_TABLES:
        _ATLAS_TABLES[key] = torch.tensor(rows, dtype=torch.float32).to(canvas.device)
    return canvas, _ATLAS_TABLES[key]


def pack_pyramid(pyr):
    """(canvas, table, levels) of a pyramid, built once per forward and shared by every readout."""
    return (*pyramid_atlas(pyr), len(pyr))


def sample_shared(packed, geom, W, base=1, li=None):
    """Each query's W x W window (cell size sig px) sampled from a packed pyramid whose finest level has stride
    2**base -> (B,Q,W*W,C), cell-major (row y, column x).

    Window level l = floor(log2 sig) reads pyramid index max(l - base, 0), i.e. stride 2**max(l, base): windows
    finer than the map interpolate it bilinearly. One grid_sample over the atlas replaces one per level: a tap
    outside its level reads the zero gap, and samples with no tap inside the level are moved off the canvas.
    li (B,Q) long: pyramid index per query instead of the one implied by the cell size (semantic stream: the level
    is chosen by the object size, the cells stay the window's).
    """
    canvas, tab, n_lvl = packed
    j = torch.arange(W, device=geom["X0"].device, dtype=torch.float32) + 0.5
    xs = geom["X0"][..., None] + j * geom["sig"][..., None]
    ys = geom["Y0"][..., None] + j * geom["sig"][..., None]
    B, C, Ha, Wa = canvas.shape
    Q = xs.shape[1]
    if li is None:
        li = (geom["lvl"] - base).clamp(0, n_lvl - 1)
    t = tab[li]
    scale = torch.exp2((li + base).float())[..., None]
    u = xs / scale - 0.5
    v = ys / scale - 0.5
    off = torch.full_like(u, -4.0)  # off the canvas: all four taps are zero padding
    u = torch.where((u > -1) & (u < t[..., 2:3]), u + t[..., 0:1], off)
    v = torch.where((v > -1) & (v < t[..., 3:4]), v + t[..., 1:2], off)
    gx = (u + 0.5) / Wa * 2 - 1
    gy = (v + 0.5) / Ha * 2 - 1
    gx, gy = torch.broadcast_tensors(gx[:, :, None, :], gy[:, :, :, None])
    grid = torch.stack([gx, gy], -1).reshape(B, Q * W, W, 2)
    out = F.grid_sample(canvas, grid.to(canvas.dtype), mode="bilinear", padding_mode="zeros", align_corners=False)
    return out.view(B, C, Q, W * W).permute(0, 2, 3, 1)  # (B,Q,W*W,C) view; the readouts contract over C


def sample_context(feat, box, stride):
    """3x3 context tokens around each box centre at the given stride -> (B,Q,9,C)."""
    off = (torch.arange(3, device=box.device, dtype=box.dtype) - 1) * stride
    t = _sample(feat, box[..., 0:1] + off, box[..., 1:2] + off, stride)  # (B,Q,C,3,3)
    return t.flatten(3).transpose(2, 3)


# ----------------------------------------------------------------------------- queries
class QueryInit(nn.Module):
    """Query embedding from the stride-4 neck feature, the dense class / objectness logits and the box."""

    def __init__(self, neck_dim, num_classes, qdim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(neck_dim + num_classes + 1 + 32, qdim), nn.GELU(), nn.Linear(qdim, qdim))
        self.norm = nn.LayerNorm(qdim)
        self.register_buffer("freqs", (2.0 ** torch.arange(4)) * math.pi, persistent=False)

    def forward(self, feat, cls_logit, obj_logit, box, img_hw):
        H, W = img_hw
        v = torch.stack([box[..., 0] / W, box[..., 1] / H,
                         torch.log(box[..., 2].clamp(min=0.5) / 8) / 4,
                         torch.log(box[..., 3].clamp(min=0.5) / 8) / 4], -1)
        ang = v[..., None] * self.freqs
        pe = torch.cat([ang.sin(), ang.cos()], -1).flatten(-2)
        x = torch.cat([feat, cls_logit.clamp(-12, 12), obj_logit.clamp(-12, 12), pe], -1)
        return self.norm(self.mlp(x))


class Readout(nn.Module):
    """One refinement step: appearance attention over the window cells and context attention (stride-8 / 16 tokens
    + scene tokens), both with the key / value projections absorbed into the query (no per-cell K/V), then
    pixel-grounded edge distributions: each box edge is the expectation over the W + 1 cell boundaries of the window.
    """

    def __init__(self, qdim, heads, fdim, W, ctx_ch, num_classes, overflow_scale=4.0):
        super().__init__()
        self.H, self.hd, self.fd, self.W = heads, qdim // heads, fdim, W
        self.P = W // 2
        C8, C16 = ctx_ch
        self.norm_in = nn.LayerNorm(qdim)
        self.q_app = nn.Linear(qdim, qdim)
        self.wk_app = nn.Parameter(torch.randn(heads, self.hd, fdim) / math.sqrt(fdim))
        self.wv_app = nn.Parameter(torch.randn(heads, self.hd, fdim) / math.sqrt(fdim))
        self.pos_app = nn.Parameter(torch.zeros(heads, 5))
        self.mom_proj = nn.Linear(heads * 5, qdim)
        self.app_out = nn.Linear(qdim, qdim)
        self.q_ctx = nn.Linear(qdim, qdim)
        self.ln8 = nn.LayerNorm(C8)
        self.ln16 = nn.LayerNorm(C16)
        self.wk8 = nn.Parameter(torch.randn(heads, self.hd, C8) / math.sqrt(C8))
        self.wv8 = nn.Parameter(torch.randn(heads, self.hd, C8) / math.sqrt(C8))
        self.wk16 = nn.Parameter(torch.randn(heads, self.hd, C16) / math.sqrt(C16))
        self.wv16 = nn.Parameter(torch.randn(heads, self.hd, C16) / math.sqrt(C16))
        self.ctx_pos = nn.Parameter(torch.zeros(heads, 18))
        self.z_kv = nn.Linear(qdim, 2 * qdim)
        self.ctx_out = nn.Linear(qdim, qdim)
        self.merge = nn.Linear(2 * qdim, qdim)
        self.ffn = nn.Sequential(nn.LayerNorm(qdim), nn.Linear(qdim, 2 * qdim), nn.GELU(), nn.Linear(2 * qdim, qdim))
        self.norm_edge = nn.LayerNorm(qdim)
        self.edge_q = nn.Linear(qdim, 8 * fdim)
        self.edge_prior = nn.Parameter(-0.1 * torch.arange(-self.P, self.P + 1).abs().float().repeat(4, 1))
        # learned tanh residual (in cells) that lets an edge leave the window
        self.overflow_scale = float(overflow_scale)
        self.edge_overflow = nn.Linear(qdim, 4)
        nn.init.zeros_(self.edge_overflow.weight)
        nn.init.zeros_(self.edge_overflow.bias)
        # down-weights the window rows / columns outside the previous box's orthogonal span
        self.ortho_weight = nn.Parameter(torch.zeros(4, 1, 1))
        self.cls_head = nn.Linear(qdim, num_classes)
        nn.init.constant_(self.cls_head.bias, -4.6)
        # localisation quality from the edge-distribution statistics, conditioned on the query
        self.qual_head = nn.Sequential(nn.Linear(20, 32), nn.GELU(), nn.Linear(32, 1))
        self.qual_z = nn.Linear(qdim, 32)
        nn.init.zeros_(self.qual_z.weight)
        nn.init.zeros_(self.qual_z.bias)

    def forward(self, z, Fp, geom, box_prev, tok8, tok16, Zt):
        """z (B,Q,D) queries, Fp (B,Q,W*W,fd) window features, geom: window of every query, box_prev (B,Q,4) xyxy,
        tok8 / tok16 (B,Q,9,C) context tokens, Zt (B,64,D) scene tokens."""
        B, Q, D = z.shape
        H, hd, fd, W, P = self.H, self.hd, self.fd, self.W, self.P
        zn = self.norm_in(z)
        X0, Y0, sig = geom["X0"], geom["Y0"], geom["sig"]
        pc = box_ops.xyxy_to_cxcywh(box_prev)
        j = torch.arange(W, device=z.device, dtype=torch.float32) + 0.5
        # cell centres relative to the previous box, in units of its size (at least one cell)
        cell_w = torch.maximum(pc[..., 2:3], sig[..., None]).clamp(min=1)
        cell_h = torch.maximum(pc[..., 3:4], sig[..., None]).clamp(min=1)
        dx = (X0[..., None] + j * sig[..., None] - pc[..., 0:1]) / cell_w
        dy = (Y0[..., None] + j * sig[..., None] - pc[..., 1:2]) / cell_h

        # ---- appearance attention with absorbed projections
        q = self.q_app(zn).view(B, Q, H, hd)
        kq = torch.einsum("bqhd,hdc->bqhc", q, self.wk_app)
        logit = torch.einsum("bqhc,bqpc->bqhp", kq, Fp).float() / math.sqrt(hd)
        pa = self.pos_app.float()
        tx = pa[:, 0].view(1, 1, H, 1) * dx[:, :, None] + pa[:, 2].view(1, 1, H, 1) * dx[:, :, None] ** 2
        ty = pa[:, 1].view(1, 1, H, 1) * dy[:, :, None] + pa[:, 3].view(1, 1, H, 1) * dy[:, :, None] ** 2
        txy = pa[:, 4].view(1, 1, H, 1, 1) * dy[:, :, None, :, None] * dx[:, :, None, None, :]
        pos = (ty[..., :, None] + tx[..., None, :] + txy).flatten(-2)
        attn = torch.softmax(logit + pos, -1)
        agg = torch.einsum("bqhp,bqpc->bqhc", attn.to(Fp.dtype), Fp)
        u = torch.einsum("bqhc,hdc->bqhd", agg, self.wv_app).reshape(B, Q, D)
        a2 = attn.view(B, Q, H, W, W)
        ax, ay = a2.sum(-2), a2.sum(-1)
        dxe, dye = dx[:, :, None], dy[:, :, None]
        mom = torch.stack([(ax * dxe).sum(-1), (ay * dye).sum(-1), (ax * dxe ** 2).sum(-1), (ay * dye ** 2).sum(-1),
                           torch.einsum("bqhyx,bqy,bqx->bqh", a2, dy, dx)], -1)
        u_app = self.app_out(u) + self.mom_proj(mom.flatten(-2))

        # ---- context attention (stride-8 3x3, stride-16 3x3, 64 scene tokens), absorbed for stride 8 / 16
        qc = self.q_ctx(zn).view(B, Q, H, hd)
        t8, t16 = self.ln8(tok8), self.ln16(tok16)
        l8 = torch.einsum("bqhc,bqtc->bqht", torch.einsum("bqhd,hdc->bqhc", qc, self.wk8), t8)
        l16 = torch.einsum("bqhc,bqtc->bqht", torch.einsum("bqhd,hdc->bqhc", qc, self.wk16), t16)
        zk, zv = self.z_kv(Zt).view(B, -1, 2, H, hd).unbind(2)
        lz = torch.einsum("bqhd,bthd->bqht", qc, zk)
        lc = torch.cat([l8, l16, lz], -1).float() / math.sqrt(hd)
        lc = lc + F.pad(self.ctx_pos.float(), (0, lz.shape[-1]))[None, None]
        ac = torch.softmax(lc, -1)
        a8, a16, az = ac.split([9, 9, lz.shape[-1]], -1)
        v8 = torch.einsum("bqhc,hdc->bqhd", torch.einsum("bqht,bqtc->bqhc", a8.to(t8.dtype), t8), self.wv8)
        v16 = torch.einsum("bqhc,hdc->bqhd", torch.einsum("bqht,bqtc->bqhc", a16.to(t16.dtype), t16), self.wv16)
        vz = torch.einsum("bqht,bthd->bqhd", az.to(zv.dtype), zv)
        u_ctx = self.ctx_out((v8 + v16 + vz).reshape(B, Q, D))

        z = z + self.merge(torch.cat([u_app, u_ctx], -1))
        z = z + self.ffn(z)

        # ---- pixel-grounded edge distributions (fp32)
        ze = self.norm_edge(z)
        coef = self.edge_q(ze).view(B, Q, 8, fd)
        proj = torch.einsum("bqkc,bqpc->bqkp", coef, Fp).float().reshape(B * Q, 8, W, W) / math.sqrt(2 * fd)
        px = F.pad(proj[:, :4], (1, 1, 0, 0), mode="replicate")  # boundary b uses cells b - 1 and b
        py = F.pad(proj[:, 4:], (0, 0, 1, 1), mode="replicate")
        sL = px[:, 0, :, : W + 1] + px[:, 1, :, 1:]
        sR = px[:, 2, :, : W + 1] + px[:, 3, :, 1:]
        sT = (py[:, 0, : W + 1, :] + py[:, 1, 1:, :]).transpose(1, 2)
        sB = (py[:, 2, : W + 1, :] + py[:, 3, 1:, :]).transpose(1, 2)
        s = torch.stack([sL, sR, sT, sB], 1).view(B, Q, 4, W, W + 1)
        org = torch.stack([X0, X0, Y0, Y0], -1)
        prev_e = torch.stack([box_prev[..., 0], box_prev[..., 2], box_prev[..., 1], box_prev[..., 3]], -1).float()
        pe_cell = (prev_e - org) / sig[..., None]
        # rows / columns outside the previous box's orthogonal span count less in the row log-sum-exp
        o_min = torch.stack([pe_cell[..., 2], pe_cell[..., 2], pe_cell[..., 0], pe_cell[..., 0]], -1)[..., None]
        o_max = torch.stack([pe_cell[..., 3], pe_cell[..., 3], pe_cell[..., 1], pe_cell[..., 1]], -1)[..., None]
        d_ortho = ((F.relu(o_min - j) + F.relu(j - o_max)) / (o_max - o_min).clamp(min=1.0)).clamp(max=4.0)
        s = s - F.softplus(self.ortho_weight.float()) * d_ortho[..., None]
        bvals = torch.arange(W + 1, device=z.device, dtype=torch.float32)
        off = (bvals - pe_cell[..., None]).clamp(-P, P) + P
        lo = off.floor().long().clamp(0, 2 * P - 1)
        fr = off - lo.float()
        tab = self.edge_prior.float()[None, None].expand(B, Q, 4, 2 * P + 1)
        prior = tab.gather(-1, lo) * (1 - fr) + tab.gather(-1, lo + 1) * fr
        # log marginal over the boundaries of the joint softmax over (row, boundary): the prior does not depend on the
        # row, so it is a softmax over the W + 1 boundaries of the row log-sum-exp
        mlogp = torch.log_softmax(torch.logsumexp(s, -2) + prior, -1)
        Pm = mlogp.exp()
        ev = (Pm * bvals).sum(-1)
        # eps keeps d sqrt / d var finite when a distribution collapses to a single boundary
        std = (((Pm * bvals ** 2).sum(-1) - ev ** 2).clamp(min=0) + 1e-6).sqrt()
        ev = ev + self.overflow_scale * torch.tanh(self.edge_overflow(ze).float())
        edges = org + sig[..., None] * ev
        box = torch.stack([edges[..., 0], edges[..., 2], edges[..., 1], edges[..., 3]], -1)
        overflow = torch.stack([Pm[..., 0, 0], Pm[..., 1, W], Pm[..., 2, 0], Pm[..., 3, W]], -1).amax(-1)
        stats = torch.cat([Pm.topk(4, -1).values, (std / W)[..., None]], -1).flatten(-2)
        stats = stats.to(self.qual_head[0].weight.dtype)
        qh = F.gelu(self.qual_head[0](stats) + self.qual_z(ze.to(stats.dtype)))
        qual = self.qual_head[2](qh).squeeze(-1).float()
        return {"z": z, "box": box, "edge_logp": mlogp, "cls": self.cls_head(z).float(),
                "qual": qual, "overflow": overflow}


class KNNSelfAttention(nn.Module):
    """Each query attends to its k nearest queries (same group: refined or denoising) with a learned geometric bias."""

    def __init__(self, qdim, heads, k):
        super().__init__()
        self.H, self.hd, self.k = heads, qdim // heads, k
        self.norm = nn.LayerNorm(qdim)
        self.qkv = nn.Linear(qdim, 3 * qdim)
        self.geo = nn.Sequential(nn.Linear(4, 32), nn.GELU(), nn.Linear(32, heads))
        self.proj = nn.Linear(qdim, qdim)
        self.ffn = nn.Sequential(nn.LayerNorm(qdim), nn.Linear(qdim, 2 * qdim), nn.GELU(), nn.Linear(2 * qdim, qdim))

    def forward(self, z, box, valid, group):
        B, Q, D = z.shape
        H, hd = self.H, self.hd
        k = min(self.k, Q)
        c = box[..., :2].float()
        d = torch.cdist(c, c)
        bad = (~valid[:, None, :]) | (group[:, :, None] != group[:, None, :])
        bad &= ~torch.eye(Q, dtype=torch.bool, device=z.device)[None]
        d = d.masked_fill(bad, float("inf"))
        dist, idx = d.topk(k, dim=-1, largest=False)
        ok = torch.isfinite(dist)
        qkv = self.qkv(self.norm(z)).view(B, Q, 3, H, hd)
        qh, kh, vh = qkv.unbind(2)
        flat = idx.reshape(B, Q * k, 1)
        kg = kh.reshape(B, Q, H * hd).gather(1, flat.expand(-1, -1, H * hd)).view(B, Q, k, H, hd)
        vg = vh.reshape(B, Q, H * hd).gather(1, flat.expand(-1, -1, H * hd)).view(B, Q, k, H, hd)
        bf = box.float()
        nb = bf.gather(1, flat.expand(-1, -1, 4)).view(B, Q, k, 4)
        scale = torch.sqrt((bf[..., 2] * bf[..., 3]).clamp(min=1.0))[..., None]
        geo = torch.stack([(nb[..., 0] - bf[..., None, 0]) / scale, (nb[..., 1] - bf[..., None, 1]) / scale,
                           torch.log(nb[..., 2].clamp(min=0.5) / bf[..., None, 2].clamp(min=0.5)),
                           torch.log(nb[..., 3].clamp(min=0.5) / bf[..., None, 3].clamp(min=0.5))], -1)
        logit = torch.einsum("bqhd,bqkhd->bqhk", qh, kg).float() / math.sqrt(hd)
        logit = logit + self.geo(geo).float().permute(0, 1, 3, 2)
        logit = logit.masked_fill(~ok[:, :, None, :], float("-inf"))
        a = torch.softmax(logit, -1)
        out = torch.einsum("bqhk,bqkhd->bqhd", a.to(vg.dtype), vg).reshape(B, Q, D)
        z = z + self.proj(out)
        return z + self.ffn(z)


def build_dn(targets, n_dn, device, bucket=1, num_classes=8):
    """Noised GT boxes as extra queries (training only, DN-DETR style).

    At most n_dn GTs per image; in crowded images the GTs are drawn with inverse-sqrt class-frequency weights, so
    rare classes are not drowned out by vehicles. The block is padded with invalid slots (ignored by the KNN
    attention and the losses) to a multiple of ``bucket``, so the query count takes few distinct shapes.
    Returns (boxes (B,nd,4) cxcywh, valid (B,nd), gt index (B,nd)) or None.
    """
    counts = [min(int(t["boxes"].shape[0]), n_dn) for t in targets]
    nd = max(counts) if counts else 0
    if nd == 0:
        return None
    if bucket > 1:
        nd = min(-(-nd // bucket) * bucket, max(n_dn, nd))
    B = len(targets)
    boxes = torch.zeros(B, nd, 4, device=device)
    boxes[..., 2:] = 8.0
    valid = torch.zeros(B, nd, dtype=torch.bool, device=device)
    gt_idx = torch.full((B, nd), -1, dtype=torch.long, device=device)
    C = max(int(num_classes), 1)
    for b, t in enumerate(targets):
        n = counts[b]
        if n == 0:
            continue
        G = int(t["boxes"].shape[0])
        if G > n:
            oh = F.one_hot(t["labels"].clamp(0, C - 1), C).float()
            cnt = (oh @ oh.sum(0)).clamp(min=1.0)
            sel = ((0.5 + torch.rand(G, device=device)) / cnt.sqrt()).topk(n, largest=True).indices
        else:
            sel = torch.randperm(G, device=device)[:n]
        g = box_ops.xyxy_to_cxcywh(t["boxes"][sel].float())
        amp = torch.maximum(0.25 * g[:, 2:].amax(-1, keepdim=True), torch.ones_like(g[:, :1]))
        shift = (torch.rand(n, 2, device=device) * 2 - 1) * amp
        scale = torch.exp((torch.rand(n, 2, device=device) * 2 - 1) * 0.3)
        boxes[b, :n] = torch.cat([g[:, :2] + shift, (g[:, 2:] * scale).clamp(min=1.0)], -1)
        valid[b, :n] = True
        gt_idx[b, :n] = sel
    return boxes, valid, gt_idx


# ----------------------------------------------------------------------------- detector
class CareTiny(nn.Module):
    """The detector. ``cfg`` is the full config (only ``cfg["model"]`` is read).

    forward(images, targets=None, k_train=None, k_override=None):
        images  (B,3,H,W) uint8 or float in [0, 255], H and W multiples of 32
        targets training only: list of {"boxes": (G,4) xyxy, "labels": (G,)} (denoising queries)
        k_train training K; k_override fixes K at inference (benchmarks)
    returns the raw outputs consumed by ``criterion.Criterion`` and ``postprocess``.
    """

    def __init__(self, cfg):
        super().__init__()
        m = dict(cfg["model"])
        self.cfg = m
        levels = int(m["backbone_levels"])
        self.backbone = Backbone(m["backbone"], m["pretrained"], m["freeze_bn"], m["freeze_stem"], levels)
        ch = self.backbone.channels
        D, Qd, C = m["neck_dim"], m["query_dim"], m["num_classes"]
        self.num_classes = C
        self.neck = FPNNeck(ch, D, Qd, m["heads"], tuple(m["neck_blocks"]), int(m["scene_layers"]))
        self.head = DecoupledDenseHead(D, C, m["head_convs"])
        self.qinit = QueryInit(D, C, Qd)
        fd = int(m["hires_dim"])
        self.hires = HiResMap(self.backbone.s2_channels, D, fd, int(m["hires_blocks"]), int(m["hires_kernel"]))
        # stride-2 residual on the dense outputs from F2 (zero init): the stride-4 head shares one feature vector
        # between 4 sub-slots, F2 sees each sub-slot's own pixels
        self.hires_head = nn.Conv2d(fd, 1 + C + 4, 1)
        nn.init.zeros_(self.hires_head.weight)
        nn.init.zeros_(self.hires_head.bias)
        self.pix = PixelMap(int(m["pix_dim"]))
        self.sem_dim = int(m["sem_dim"])
        self.sem_proj = nn.ModuleList([nn.Sequential(nn.Conv2d(D, self.sem_dim, 1),
                                                     LayerNorm2d(self.sem_dim, keep_dtype=True))
                                       for _ in range(len(ch))])
        fd = fd + int(m["pix_dim"]) + self.sem_dim
        # stride-8 / 16 backbone features projected once (1x1 conv + LN) and shared by every readout
        cd = int(m["ctx_dim"]) or Qd
        self.ctx_proj = nn.ModuleList([nn.Sequential(nn.Conv2d(c, cd, 1), LayerNorm2d(cd, keep_dtype=True))
                                       for c in (ch[1], ch[2])])
        # frame shift (dcx, dcy, dlog w, dlog h) of a re-centred window, added to the query (zero init)
        self.recenter_pe = nn.Sequential(nn.Linear(4, Qd), nn.GELU(), nn.Linear(Qd, Qd))
        nn.init.zeros_(self.recenter_pe[-1].weight)
        nn.init.zeros_(self.recenter_pe[-1].bias)
        self.select_score_alpha = float(m["select_score_alpha"])
        self.recenter_sig_ema = float(m["recenter_sig_ema"])
        self.readouts = nn.ModuleList(
            [Readout(Qd, m["heads"], fd, m["window"], (cd, cd), C, float(m["edge_overflow_scale"]))
             for _ in range(int(m["num_readouts"]))])
        self.knn = KNNSelfAttention(Qd, m["heads"], m["knn"])
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1) * 255, persistent=False)
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1) * 255, persistent=False)
        self.dn_per_image = 0  # set by the trainer
        self.dn_bucket = 1
        self.channels_last = False

    def k_from_count(self, n_hat, M):
        """Inference K: the smallest bucket >= k_scale_eval * n_hat + k_offset."""
        b = torch.tensor(self.cfg["k_buckets"], device=n_hat.device, dtype=torch.float32)
        scale = self.cfg["k_scale_eval"] or self.cfg["k_scale"]
        need = scale * n_hat.float() + self.cfg["k_offset"]
        idx = torch.searchsorted(b, need.contiguous()).clamp(max=len(b) - 1)
        return b[idx].long().clamp(max=M)

    def forward(self, images, targets=None, k_train=None, k_override=None):
        m = self.cfg
        W, L = m["window"], m["pyramid_levels"]
        B, _, Hi, Wi = images.shape
        x = (images.float() - self.mean) / self.std
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        feats = self.backbone(x)
        H4, Z, P = self.neck(feats, F.pixel_unshuffle(x, 4))
        obj, cls, box_raw = self.head(H4)
        s2, self.backbone.s2 = self.backbone.s2, None  # do not keep the activation alive between steps
        F2 = self.hires(s2, x, H4)
        r = self.hires_head(F2)
        C = self.num_classes
        obj, cls, box_raw = obj + r[:, :1], cls + r[:, 1:1 + C], box_raw + r[:, 1 + C:]
        Hs, Ws = obj.shape[-2:]
        dense_boxes = decode_dense(box_raw)
        obj_f = obj.float().flatten(1)
        cls_f = cls.float().flatten(2).transpose(1, 2)
        M = min(m["num_candidates"], Hs * Ws)

        # ---- candidate routing
        a = self.select_score_alpha
        route_logit = (1.0 - a) * obj_f.detach() + a * cls_f.detach().amax(-1)
        n_hat = count_objects(obj_f.detach(), Hs, Ws, m["count_thr"])
        kmax = None  # known on the host for a fixed K: no .item() sync in select_candidates
        if k_override is not None:
            kmax = min(max(int(k_override), 1), M)
            k_list = torch.full((B,), kmax, device=x.device, dtype=torch.long)
        elif self.training:
            kmax = min(max(int(k_train), 1), M)
            k_list = torch.full((B,), kmax, device=x.device, dtype=torch.long)
        else:
            k_list = self.k_from_count(n_hat, M).clamp(min=1, max=M)
        candM, kidx, active = select_candidates(route_logit, Hs, Ws, M, m["quota_regions"], m["quota_per_region"],
                                                k_list, m["spatial_frac"], kmax, float(m["region_min_prob"]))
        ridx = candM.gather(1, kidx)
        Kb = ridx.shape[1]
        qbox = dense_boxes.gather(1, ridx[..., None].expand(-1, -1, 4)).detach()
        qcls = cls_f.gather(1, ridx[..., None].expand(-1, -1, C)).detach()
        qobj = obj_f.gather(1, ridx)[..., None].detach()
        valid = active.clone()
        group = torch.zeros_like(ridx)
        dn_gt = None
        if self.training and targets is not None and self.dn_per_image > 0:
            dn = build_dn(targets, self.dn_per_image, x.device, self.dn_bucket, C)
            if dn is not None:
                dbox, dvalid, dn_gt = dn
                sx = (dbox[..., 0] / 2).floor().clamp(0, Ws - 1).long()
                sy = (dbox[..., 1] / 2).floor().clamp(0, Hs - 1).long()
                didx = sy * Ws + sx
                qbox = torch.cat([qbox, dbox], 1)
                qcls = torch.cat([qcls, cls_f.gather(1, didx[..., None].expand(-1, -1, C)).detach()], 1)
                qobj = torch.cat([qobj, obj_f.gather(1, didx)[..., None].detach()], 1)
                valid = torch.cat([valid, dvalid], 1)
                group = torch.cat([group, torch.ones_like(didx)], 1)
        qbox = torch.cat([qbox[..., :2], qbox[..., 2:].clamp(min=1.0)], -1)

        # ---- queries, context tokens and the shared maps of the windows
        h4c = _sample(H4.float(), qbox[..., 0:1], qbox[..., 1:2], 4)[..., 0, 0]
        z0 = self.qinit(h4c, qcls, qobj, qbox, (Hi, Wi))
        geom = window_geometry(qbox, W, m["window_obj"], L)
        cf8 = self.ctx_proj[0](feats[1]).float()
        cf16 = self.ctx_proj[1](feats[2]).float()
        tok8 = sample_context(cf8, qbox, 8)
        tok16 = sample_context(cf16, qbox, 16)
        # F2 is fp32 (LayerNorm / residual path); the pixel and semantic maps leave their LayerNorm in the autocast
        # dtype and are sampled in fp32, so the sampling grid is not rounded to bf16 (taps off by up to ~1.2 px)
        sem_lvls = [proj(p).float() for proj, p in zip(self.sem_proj, P)]
        for lvl in range(len(sem_lvls) - 2, -1, -1):  # top-down fusion: every PAN level reaches the small objects
            sem_lvls[lvl] = sem_lvls[lvl] + 0.5 * F.interpolate(
                sem_lvls[lvl + 1], size=sem_lvls[lvl].shape[-2:], mode="bilinear", align_corners=False)
        maps = {"f2": pack_pyramid(build_pyramid(F2, L - 1)),  # strides 2 .. 2**(L-1)
                "pix": pack_pyramid(build_pyramid(self.pix(x).float(), L)),  # strides 1 .. 2**(L-1)
                "sem": pack_pyramid(sem_lvls)}
        Fm = self._window_feats(maps, geom, qbox, x)

        # ---- readout cascade
        bprev = box_ops.cxcywh_to_xyxy(qbox)
        prev_cb = qbox
        z, refine, n_ro = z0, [], len(self.readouts)
        for i, ro in enumerate(self.readouts):
            if i > 0:
                # re-centre the window on the previous readout's box and sample its features again
                bprev = box_ops.sort_xyxy(refine[-1]["box"]).detach()
                cb = box_ops.xyxy_to_cxcywh(bprev)
                cb = torch.cat([cb[..., :2], cb[..., 2:].clamp(min=1.0)], -1)
                pwh = prev_cb[..., 2:].clamp(min=1.0)
                delta = torch.cat([(cb[..., :2] - prev_cb[..., :2]) / pwh, torch.log(cb[..., 2:] / pwh)], -1)
                z = z + self.recenter_pe(delta.clamp(-4.0, 4.0).to(z.dtype))
                if self.recenter_sig_ema > 0:
                    # the window shrinks at most halfway towards the new box and grows with the edge overflow
                    s_ema = self.recenter_sig_ema
                    wh_floor = s_ema * prev_cb[..., 2:] + (1.0 - s_ema) * cb[..., 2:]
                    ov_grow = 1.0 + s_ema * refine[-1]["overflow"].detach()[..., None]
                    w_cb = torch.cat([cb[..., :2], torch.maximum(cb[..., 2:], wh_floor) * ov_grow], -1)
                else:
                    w_cb = cb
                prev_cb = w_cb
                geom = window_geometry(w_cb, W, m["window_obj"], L)
                Fm = self._window_feats(maps, geom, w_cb, x)
                tok8, tok16 = sample_context(cf8, cb, 8), sample_context(cf16, cb, 16)
                if i == n_ro - 1:  # the one-to-one readout: queries first see their k nearest neighbours
                    z = self.knn(z, box_ops.xyxy_to_cxcywh(bprev), valid, group)
            r = ro(z, Fm, geom, bprev, tok8, tok16, Z)
            r["geom"] = geom  # the window this readout read (frame of its edge-distribution target)
            refine.append(r)
            z = r["z"]
        return {
            "dense": {"obj": obj_f, "cls": cls_f, "boxes": dense_boxes, "hw": (Hs, Ws)},
            "q": {"ridx": ridx, "candM": candM, "kidx": kidx, "active": active, "valid": valid, "group": group,
                  "geom": refine[0]["geom"], "Kb": Kb, "dn_gt": dn_gt},
            "refine": refine,
            "n_hat": n_hat,
            "k_list": k_list,
        }

    def _window_feats(self, maps, geom, box, x):
        """Window features (B,Q,W*W,fd) of the windows ``geom`` (of the cxcywh boxes ``box``): F2 detail, stride-1
        pixels and PAN semantics (level chosen by the box size), concatenated per cell."""
        W = self.cfg["window"]
        Fm = sample_shared(maps["f2"], geom, W, base=1)
        Fm = torch.cat([Fm, sample_shared(maps["pix"], geom, W, base=0).to(Fm.dtype)], -1)
        ls = semantic_level(box, float(self.cfg["sem_obj_cells"]), maps["sem"][2])
        Fm = torch.cat([Fm, sample_shared(maps["sem"], geom, W, base=2, li=ls).to(Fm.dtype)], -1)
        if torch.is_autocast_enabled(x.device.type):  # one cast here instead of one per readout einsum
            Fm = Fm.to(torch.get_autocast_dtype(x.device.type))
        return Fm.contiguous()


def compile_hot_modules(model, mode="default", backbone=False):
    """torch.compile (in place: same parameters and state_dict keys) the modules dominated by small memory-bound ops
    (LayerNorms, casts, elementwise chains), which inductor fuses; convs and matmuls stay on cuDNN / cuBLAS. K and
    the denoising count change the shapes (a handful of buckets), so the recompile limit is raised."""
    import torch._dynamo
    cfg = torch._dynamo.config
    for name in ("cache_size_limit", "recompile_limit"):
        if hasattr(cfg, name):
            setattr(cfg, name, max(getattr(cfg, name), 64))
    for m in (model.neck, model.head, model.hires, model.pix, *model.sem_proj, *model.ctx_proj, model.recenter_pe,
              model.knn, *model.readouts):
        m.compile(mode=mode)
    if backbone:
        # fuses timm's frozen-BN epilogues (fp32 scale / bias) into the conv kernels; the stride-2 stem hook is
        # registered on the inner stem1 module, which still runs it
        model.backbone.body.compile(mode=mode)
    return model


# ----------------------------------------------------------------------------- post-processing
@torch.no_grad()
def postprocess(out, sizes, ecfg):
    """Detections of every image: refined boxes of the K refined queries + coarse boxes of the other M - K
    candidates, then class-aware NMS. sizes: (h, w) per image for clipping. Returns a list of
    {"boxes" (n,4) xyxy, "scores" (n,), "labels" (n,)}."""
    res = []
    for b in range(out["refine"][-1]["box"].shape[0]):
        rb, rs, cb, cs, rd = detection_candidates(out, b, ecfg)
        res.append(select_detections(rb, rs, cb, cs, sizes[b], ecfg, rd))
    return res


@torch.no_grad()
def detection_candidates(out, b, ecfg):
    """Image b: refined boxes (n,4) xyxy with class scores (n,C), coarse boxes (m,4) with class scores (m,C), and the
    dense class scores (n,C) of the refined queries' own sub-slots (for eval.score_fuse)."""
    r = out["refine"][-1]
    q = out["q"]
    d = out["dense"]
    Kb = q["Kb"]
    act = q["active"][b]
    rb = box_ops.sort_xyxy(r["box"][b, :Kb][act].float())
    rs = torch.sigmoid(r["cls"][b, :Kb][act]) * torch.sigmoid(r["qual"][b, :Kb][act])[:, None]
    rd = torch.sigmoid(d["cls"][b, q["ridx"][b, :Kb][act]].float())
    cand = q["candM"][b]
    keep_c = torch.ones(cand.shape[0], dtype=torch.bool, device=cand.device)
    keep_c[q["kidx"][b, :Kb][act]] = False  # a refined candidate is represented by its refined box
    cidx = cand[keep_c]
    cb = box_ops.cxcywh_to_xyxy(d["boxes"][b, cidx].float())
    cs = torch.sigmoid(d["cls"][b, cidx].float())
    p = float(ecfg["coarse_obj_pow"])
    if p > 0:  # refined scores fold the objectness in through the quality head, coarse scores through this
        cs = cs * torch.sigmoid(d["obj"][b, cidx].float())[:, None] ** p
    return rb, rs, cb, cs, rd


@torch.no_grad()
def select_detections(rb, rs, cb, cs, size, ecfg, rd=None):
    """Merge refined and coarse candidates (arg-max class each), threshold, keep the pre-NMS top, class-aware IoU
    NMS, keep max_dets and clip to the image (h, w). eval.score_fuse a > 0: refined scores rs**(1-a) * rd**a."""
    h, w = size
    a = float(ecfg["score_fuse"])
    if a > 0 and rd is not None:
        rs = rs.clamp(min=1e-9) ** (1 - a) * rd.clamp(min=1e-9) ** a
    rsc, rl = rs.max(-1)
    csc, cl = cs.max(-1)
    csc = csc * ecfg["coarse_score_mult"]
    boxes = torch.cat([rb, cb])
    scores = torch.cat([rsc, csc])
    labels = torch.cat([rl, cl])
    keep = scores > ecfg["score_thr"]
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    if scores.numel() > ecfg["pre_nms_topk"]:
        t = scores.topk(ecfg["pre_nms_topk"]).indices
        boxes, scores, labels = boxes[t], scores[t], labels[t]
    k = batched_nms(boxes, scores, labels, ecfg["nms_iou"])[: ecfg["max_dets"]]
    boxes = boxes[k]
    boxes = torch.stack([boxes[:, 0].clamp(0, w), boxes[:, 1].clamp(0, h),
                         boxes[:, 2].clamp(0, w), boxes[:, 3].clamp(0, h)], 1)
    return {"boxes": boxes, "scores": scores[k], "labels": labels[k]}
