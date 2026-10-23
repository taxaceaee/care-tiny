import random

import numpy as np
import pytest
import torch
from torch.utils.data import DistributedSampler

from care_tiny.data import (
    STRONG_OPS,
    AITODv2,
    EpochSampler,
    RepeatFactorSampler,
    apply_photometric,
    collate,
    hflip,
    parse_stage2_keep,
    photometric_batch,
    photometric_params,
    repeat_factors,
    rot90,
    vflip,
)

from conftest import small_cfg


def _check_box_follows_pixels(fn):
    img = torch.zeros(3, 40, 60, dtype=torch.uint8)
    box = torch.tensor([[5.0, 7.0, 15.0, 12.0]])
    img[:, 7:12, 5:15] = 255
    img2, b2 = fn(img, box)
    ys, xs = torch.nonzero(img2[0] > 0, as_tuple=True)
    got = torch.tensor([[xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]], dtype=torch.float32)
    assert torch.allclose(got, b2), (got, b2)


def test_flips_and_rot90_move_boxes_with_their_pixels():
    _check_box_follows_pixels(hflip)
    _check_box_follows_pixels(vflip)
    for k in (1, 2, 3):
        _check_box_follows_pixels(lambda i, b, k=k: rot90(i, b, k))


def test_copy_paste_pastes_source_pixels_without_covering_gt():
    """Image 0 (grey 10) holds common-class boxes; image 1 holds 3 rare boxes painted 200 + k on grey 50. Pasted
    boxes keep their size, sit on their own source pixels and never intersect a box present before the paste."""
    ds = AITODv2.__new__(AITODv2)
    ds.cat_ids = [0, 1]
    common = np.array([[10, 10, 8, 8], [60, 60, 8, 8], [100, 20, 6, 6]], dtype=np.float32)
    rare = np.array([[20, 20, 6, 4], [70, 30, 5, 5], [40, 90, 8, 6]], dtype=np.float32)
    ds.boxes = [common, rare] + [common] * 38  # class 1 in 1 of 40 images: rare
    ds.labels = [np.zeros(3, np.int64), np.ones(3, np.int64)] + [np.zeros(3, np.int64)] * 38
    ds.images = [None] * 40
    ds.aug = {"copy_paste_p": 1.0, "copy_paste_img_frac": 0.05, "copy_paste_max": 8}
    src = torch.full((3, 128, 128), 50, dtype=torch.uint8)
    for k, (x, y, w, h) in enumerate(rare.astype(int)):
        src[:, y:y + h, x:x + w] = 200 + k
    ds.load_image = lambda j: src.clone() if j == 1 else torch.full((3, 128, 128), 10, dtype=torch.uint8)
    ds.cp_bank = ds._build_cp_bank()
    assert list(ds.cp_bank) == [1]
    random.seed(0)
    for _ in range(10):
        img, b, lab = ds._raw(0)
        out, nb, nl = ds._copy_paste(img, b, lab)
        assert torch.equal(nb[:3], b) and torch.equal(nl[:3], lab)
        new, newl = nb[3:], nl[3:]
        assert new.shape[0] > 0 and bool((newl == 1).all())
        for bb in new:
            x1, y1, x2, y2 = [int(round(float(v))) for v in bb]
            vals = out[0, y1:y2, x1:x2].unique()
            assert vals.numel() == 1 and int(vals[0]) >= 200  # a rare object's own pixels
            inter = (b[:, 0] < bb[2]) & (b[:, 2] > bb[0]) & (b[:, 1] < bb[3]) & (b[:, 3] > bb[1])
            assert not bool(inter.any())
        assert torch.equal(out[:, 10:18, 10:18], img[:, 10:18, 10:18])  # GT pixels untouched


def _bare_dataset(strong, keep, aug):
    ds = AITODv2.__new__(AITODv2)
    ds.aug = {"hflip_p": 0.0, "vflip_p": 0.0, "photometric_p": 0.0, "scale_p": 0.0, "rot90_p": 0.0,
              "stage2_keep": keep, **aug}
    ds.strong, ds.gpu_photometric = strong, False
    ds.stage2_keep = parse_stage2_keep(ds.aug)
    return ds


def test_stage2_keep_selects_the_ops_of_the_no_aug_stage():
    ds = _bare_dataset(False, ["copy_paste"], {})
    assert ds.op_on("copy_paste") and not any(ds.op_on(o) for o in STRONG_OPS - {"copy_paste"})
    assert all(_bare_dataset(True, [], {}).op_on(o) for o in STRONG_OPS)
    with pytest.raises(ValueError):
        parse_stage2_keep({"stage2_keep": ["hflip"]})
    img = torch.arange(3 * 20 * 30).remainder(251).to(torch.uint8).reshape(3, 20, 30)
    box, lab = torch.tensor([[2.0, 3.0, 8.0, 9.0]]), torch.tensor([1])
    random.seed(0)
    out, *_ = _bare_dataset(False, [], {"rot90_p": 1.0})._augment(img, box, lab)
    assert torch.equal(out, img)  # dropped in the no-aug stage
    random.seed(0)
    out, *_ = _bare_dataset(False, ["rot90"], {"rot90_p": 1.0})._augment(img, box, lab)
    assert not (out.shape == img.shape and torch.equal(out, img))  # kept: rotated


def test_gpu_photometric_matches_per_image_ops_and_keeps_borders_black():
    random.seed(0)
    aug = dict(small_cfg()["data"]["aug"])
    img = torch.randint(0, 255, (3, 40, 56), dtype=torch.uint8)
    ops = photometric_params(aug)
    canvas = torch.zeros(1, 3, 48, 64, dtype=torch.uint8)
    canvas[0, :, 3:43, 5:61] = img
    photometric_batch(canvas, [{"photo": ops, "valid": (5, 3, 61, 43)}])
    assert torch.equal(canvas[0, :, 3:43, 5:61], apply_photometric(img, ops))
    assert int(canvas.sum()) == int(canvas[0, :, 3:43, 5:61].sum())


def test_epoch_sampler_order_and_skip():
    data = list(range(29))
    for world in (1, 2):
        for r in range(world):
            s = EpochSampler(data, world, r, 42)
            ref = DistributedSampler(data, num_replicas=world, rank=r, shuffle=True, seed=42, drop_last=True)
            s.set_epoch(5)
            ref.set_epoch(5)
            assert list(s) == list(ref)
            s.skip = 4
            assert list(s) == list(ref)[4:] and len(s) == len(ref) - 4


def test_repeat_factor_sampler():
    class DS:
        cat_ids = [0, 1]
        labels = [np.array([0])] * 98 + [np.array([1]), np.array([0, 1])]  # class 1 in 2% of the images

    r = repeat_factors(DS(), 0.08)
    assert np.allclose(r[:98], 1.0) and r[98] == pytest.approx(2.0) and r[99] == pytest.approx(2.0)
    for world in (1, 2):
        shards = []
        for rank in range(world):
            s = RepeatFactorSampler(r, world, rank, 3)
            s.set_epoch(4)
            a = list(s)
            assert a == list(s) and len(a) == 102 // world  # deterministic per epoch, fixed length
            s.skip = 5
            assert list(s) == a[5:]
            shards += a
        counts = np.bincount(shards, minlength=100)
        assert counts[98] == 2 and counts[99] == 2


def test_collate_pads_to_one_static_shape():
    imgs = [torch.ones(3, 801, 800, dtype=torch.uint8), torch.ones(3, 640, 700, dtype=torch.uint8)]
    t = [{"boxes": torch.tensor([[790.0, 795.0, 800.0, 801.0], [1.0, 1.0, 5.0, 5.0]]), "labels": torch.tensor([0, 1]),
          "orig_size": (801, 800)}, {"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.long),
                                     "orig_size": (640, 700)}]
    batch, targets = collate(list(zip(imgs, t)), 32, 800)
    assert batch.shape == (2, 3, 800, 800)
    assert targets[0]["boxes"].tolist() == [[790.0, 795.0, 800.0, 800.0], [1.0, 1.0, 5.0, 5.0]]
    batch, _ = collate([(imgs[1], t[1])], 32, 0)
    assert batch.shape == (1, 3, 640, 704)


def test_dataset_reads_the_official_layout(aitod_root):
    cfg = small_cfg()
    test = AITODv2(aitod_root, "test")
    img, t = test[0]
    assert img.dtype == torch.uint8 and img.shape == (3, 128, 128) and t["orig_size"] == (128, 128)
    assert test.cat_ids == [1, 2, 3] and t["labels"].max() <= 2  # official ids 1.. -> labels 0..
    train = AITODv2(aitod_root, "trainval", train=True, aug=dict(cfg["data"]["aug"]), gpu_photometric=True)
    random.seed(1)
    img, t = train[1]
    assert set(t) >= {"boxes", "labels", "photo", "valid"} and t["boxes"].shape[0] == t["labels"].shape[0]
