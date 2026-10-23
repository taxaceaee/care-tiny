import torch

from care_tiny.config import load_config
from care_tiny.criterion import Criterion
from care_tiny.model import (
    CareTiny,
    Readout,
    _sample,
    build_dn,
    build_pyramid,
    decode_dense,
    pack_pyramid,
    postprocess,
    sample_shared,
    select_candidates,
    window_geometry,
)

from conftest import small_cfg, synthetic_batch


def test_released_architecture_size():
    model = CareTiny(load_config(None, ["model.pretrained=false"]))
    assert sum(p.numel() for p in model.parameters()) == 10_024_663
    assert sum(p.numel() for p in model.backbone.parameters()) == 5_999_930  # HGNetv2-B2


def test_subslot_decode_geometry():
    """A zero regression puts each sub-slot's box centre in the middle of its 2 x 2 px cell, 8 px wide."""
    b = decode_dense(torch.zeros(1, 4, 3, 5))
    cx, cy = b[0, :, 0].view(3, 5), b[0, :, 1].view(3, 5)
    assert torch.allclose(cx, 2 * torch.arange(5.0)[None].expand(3, 5) + 1)
    assert torch.allclose(cy, 2 * torch.arange(3.0)[:, None].expand(3, 5) + 1)
    assert torch.allclose(b[0, :, 2:], torch.full((15, 2), 8.0))


def _sample_levels_reference(pyr, geom, W, base):
    """One bilinear grid_sample per pyramid level, selected per query (what the atlas sampling must equal)."""
    j = torch.arange(W, dtype=torch.float32) + 0.5
    xs = geom["X0"][..., None] + j * geom["sig"][..., None]
    ys = geom["Y0"][..., None] + j * geom["sig"][..., None]
    li = (geom["lvl"] - base).clamp(0, len(pyr) - 1)
    out = None
    for i, P in enumerate(pyr):
        s = _sample(P, xs, ys, 2 ** (i + base))  # (B,Q,C,W,W)
        sel = (li == i)[..., None, None, None]
        out = torch.where(sel, s, torch.zeros_like(s) if out is None else out)
    return out.flatten(3).transpose(2, 3)


def test_atlas_sampling_equals_per_level_sampling():
    g = torch.Generator().manual_seed(0)
    fmap = torch.randn(2, 5, 50, 70, generator=g)
    pyr = build_pyramid(fmap, 4)
    box = torch.cat([torch.rand(2, 30, 2, generator=g) * 120, torch.rand(2, 30, 2, generator=g) * 90 + 1], -1)
    geom = window_geometry(box, 8, 6.0, 5)
    for base in (0, 1):
        got = sample_shared(pack_pyramid(pyr), geom, 8, base=base)
        ref = _sample_levels_reference(pyr, geom, 8, base)
        assert torch.allclose(got, ref, atol=1e-4)  # float rounding of the grid normalisation


def test_native_window_reads_exact_pixels():
    """Objects up to window_obj px get native cells: the stride-1 window samples the pixels themselves."""
    img = torch.rand(1, 3, 64, 64)
    geom = window_geometry(torch.tensor([[[30.0, 21.0, 5.0, 4.0]]]), 8, 6.0, 3)
    assert float(geom["sig"]) == 1.0 and (float(geom["X0"]), float(geom["Y0"])) == (26.0, 17.0)
    win = sample_shared(pack_pyramid(build_pyramid(img, 3)), geom, 8, base=0)[0, 0]  # (64, 3) cell-major
    assert torch.allclose(win.T.reshape(3, 8, 8), img[0, :, 17:25, 26:34], atol=1e-4)


def test_select_candidates_keeps_region_winners_and_k():
    torch.manual_seed(0)
    Hs = Ws = 32
    obj = torch.randn(2, Hs * Ws)
    k = torch.tensor([16, 32])
    candM, kidx, active = select_candidates(obj, Hs, Ws, 128, 4, 4, k, 0.25)
    assert candM.shape == (2, 128)
    for b in range(2):
        assert len(set(candM[b].tolist())) == 128
        assert int(active[b].sum()) == int(k[b])
    reg = torch.sigmoid(obj[0]).view(4, 8, 4, 8).permute(0, 2, 1, 3).reshape(16, 64)
    top = reg.argmax(1)
    for r in range(16):
        gy, gx = (r // 4) * 8 + top[r] // 8, (r % 4) * 8 + top[r] % 8
        assert int(gy * Ws + gx) in set(candM[0].tolist())


def test_k_pick_gives_every_object_its_peak():
    """Two blobs, the strong one with many high sub-slots: with K = 2 each blob gets its own peak (a plain top-2
    would spend both refined queries on the strong blob)."""
    Hs = Ws = 16
    yy, xx = torch.meshgrid(torch.arange(Hs).float(), torch.arange(Ws).float(), indexing="ij")
    p = 0.9 * torch.exp(-((yy - 4) ** 2 + (xx - 4) ** 2) / 8) + 0.5 * torch.exp(-((yy - 11) ** 2 + (xx - 11) ** 2) / 2)
    obj = torch.logit(p.clamp(1e-4, 1 - 1e-4)).view(1, -1)
    candM, kidx, _ = select_candidates(obj, Hs, Ws, 64, 1, 1, torch.tensor([2]), 0.0, 2)
    assert set(candM[0, kidx[0]].tolist()) == {4 * Ws + 4, 11 * Ws + 11}


def test_inference_k_buckets():
    model = CareTiny(small_cfg(["model.k_scale_eval=1.0"]))
    assert model.k_from_count(torch.tensor([0, 10, 1000]), 256).tolist() == [16, 32, 64]  # need = n + 8


def test_edge_readout_places_edges_on_feature_boundaries():
    """Window features with a step at known columns / rows: the edge expectations land on those boundaries."""
    W, fd = 8, 4
    r = Readout(16, 2, fd, W, (4, 4), 2).double()
    with torch.no_grad():
        r.edge_q.weight.zero_()
        bias = torch.zeros(8, fd, dtype=torch.float64)
        bias[1, 0] = 1  # left edge at the column with channel 0
        bias[2, 1] = 1  # right edge one after the column with channel 1
        bias[5, 2] = 1  # top edge at the row with channel 2
        bias[6, 3] = 1  # bottom edge one after the row with channel 3
        r.edge_q.bias.copy_(bias.flatten())
        r.edge_prior.zero_()
    Fm = torch.zeros(1, fd, W, W, dtype=torch.float64)
    Fm[0, 0, :, 2] = 40
    Fm[0, 1, :, 5] = 40
    Fm[0, 2, 1, :] = 40
    Fm[0, 3, 6, :] = 40
    Fp = Fm.permute(0, 2, 3, 1).reshape(1, 1, W * W, fd)  # cell-major (row y, column x)
    geom = {"X0": torch.tensor([[10.0]], dtype=torch.float64), "Y0": torch.tensor([[20.0]], dtype=torch.float64),
            "sig": torch.tensor([[1.0]], dtype=torch.float64)}
    prev = torch.tensor([[[12.0, 21.0, 16.0, 27.0]]], dtype=torch.float64)
    out = r(torch.randn(1, 1, 16, dtype=torch.float64), Fp, geom, prev, torch.randn(1, 1, 9, 4).double(),
            torch.randn(1, 1, 9, 4).double(), torch.randn(1, 64, 16).double())
    assert torch.allclose(out["box"][0, 0], prev[0, 0], atol=0.05)


def test_denoising_queries_favour_rare_classes_in_crowded_images():
    boxes = torch.rand(300, 2) * 700
    boxes = torch.cat([boxes, boxes + 6], 1)
    labels = torch.full((300,), 5)
    labels[[7, 150, 299]] = torch.tensor([0, 7, 1])  # three rare objects among 297 vehicles
    torch.manual_seed(0)
    for _ in range(5):
        _, valid, gt_idx = build_dn([{"boxes": boxes, "labels": labels}], 50, "cpu", bucket=25)
        picked = set(gt_idx[0][valid[0]].tolist())
        assert len(picked) == 50 and {7, 150, 299} <= picked


def test_every_parameter_gets_a_gradient_and_eval_outputs_are_valid():
    torch.manual_seed(0)
    cfg = small_cfg(["model.num_classes=3"])
    model = CareTiny(cfg)
    model.dn_per_image = 8
    imgs, targets = synthetic_batch()
    model.train()
    loss, parts = Criterion(cfg)(model(imgs, targets, k_train=32), targets)
    assert torch.isfinite(loss) and all(torch.isfinite(v) for v in parts.values())
    loss.backward()
    assert [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None] == []
    model.eval()
    with torch.no_grad():
        out = model(imgs)
        res = postprocess(out, [(128, 128)] * 2, cfg["eval"])
    assert len(res) == 2
    for r in res:
        assert r["boxes"].shape[1] == 4 and r["boxes"].shape[0] == r["scores"].shape[0] <= cfg["eval"]["max_dets"]
        assert bool((r["boxes"] >= 0).all() and (r["boxes"] <= 128).all())
        assert bool((r["scores"][:-1] >= r["scores"][1:]).all())  # NMS keeps the score order

