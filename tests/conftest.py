"""Shared helpers: a small CPU configuration of the detector and synthetic AI-TOD-style data."""

import json
import os
import random
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from care_tiny.config import load_config  # noqa: E402

CLASSES = ["airplane", "bridge", "storage-tank", "ship", "swimming-pool", "vehicle", "person", "wind-mill"]

# the released architecture, scaled down (HGNetv2-B0, narrow neck and queries, 8-cell windows): ~2.4M parameters,
# a training step on 2 x 128 x 128 images takes ~0.25 s on a CPU
SMALL = ["model.backbone=hgnetv2_b0", "model.pretrained=false", "model.neck_dim=32", "model.query_dim=64",
         "model.heads=2", "model.hires_dim=16", "model.sem_dim=16", "model.pix_dim=4", "model.window=8",
         "model.window_obj=6.0", "model.num_candidates=256", "model.k_buckets=[16,32,64]", "model.k_offset=8",
         "model.quota_regions=4", "model.knn=8", "train.dn_per_image=8", "train.train_k_max=64",
         "train.compile=false", "data.num_workers=0", "data.pad_to=0", "eval.eval_procs=1", "eval.batch_size=2"]


def small_cfg(extra=()):
    return load_config(None, SMALL + list(extra))


def disjoint_boxes(n, size, lo=2.0, hi=10.0, rng=None):
    rng = rng or random.Random(0)
    out = []
    for _ in range(20000):
        if len(out) >= n:
            break
        w, h = rng.uniform(lo, hi), rng.uniform(lo, hi)
        x, y = rng.uniform(0, size - w), rng.uniform(0, size - h)
        b = (x, y, x + w, y + h)
        if all(b[2] <= o[0] or o[2] <= b[0] or b[3] <= o[1] or o[3] <= b[1] for o in out):
            out.append(b)
    return torch.tensor(out, dtype=torch.float32)


def rand_boxes(n, size=100.0, gen=None):
    xy = torch.rand(n, 2, generator=gen) * size
    wh = torch.rand(n, 2, generator=gen) * 20 + 1
    return torch.cat([xy, xy + wh], 1)


def synthetic_batch(n=2, size=128, seed=0, num_classes=3, boxes_per_image=8):
    """Images with coloured boxes (one colour per class) on a noisy background, and their targets."""
    g = torch.Generator().manual_seed(seed)
    imgs = torch.randint(60, 90, (n, 3, size, size), dtype=torch.uint8, generator=g)
    colors = torch.tensor([[255, 40, 40], [40, 255, 40], [40, 40, 255], [255, 255, 40], [40, 255, 255],
                           [255, 40, 255], [140, 140, 140], [255, 140, 0]])
    targets = []
    for i in range(n):
        boxes = disjoint_boxes(boxes_per_image, size - 8, 3.0, 12.0, random.Random(seed + i)).round() + 4
        labels = torch.randint(0, num_classes, (len(boxes),), generator=g)
        for b, lab in zip(boxes, labels):
            x1, y1, x2, y2 = [int(v) for v in b]
            imgs[i, :, y1:y2, x1:x2] = colors[int(lab)][:, None, None].to(torch.uint8)
        targets.append({"boxes": boxes, "labels": labels, "image_id": i + 1, "orig_size": (size, size)})
    return imgs, targets


def write_aitod(root, imgs, targets, splits=("trainval", "test"), num_classes=3):
    """A tiny dataset in the official AI-TOD-v2 layout (category ids start at 1, as in the official files)."""
    from torchvision.io import encode_png, write_file
    os.makedirs(os.path.join(root, "annotations"), exist_ok=True)
    images, anns = [], []
    for i, (im, t) in enumerate(zip(imgs, targets)):
        images.append({"id": i + 1, "file_name": f"{i}.png", "width": int(im.shape[2]), "height": int(im.shape[1])})
        for b, lab in zip(t["boxes"].tolist(), t["labels"].tolist()):
            anns.append({"id": len(anns) + 1, "image_id": i + 1, "category_id": lab + 1, "iscrowd": 0,
                         "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]], "area": (b[2] - b[0]) * (b[3] - b[1])})
    for split in splits:
        d = os.path.join(root, "images", "test" if split == "test" else "trainval")
        os.makedirs(d, exist_ok=True)
        for i, im in enumerate(imgs):
            write_file(os.path.join(d, f"{i}.png"), encode_png(im))
        with open(os.path.join(root, "annotations", f"aitodv2_{split}.json"), "w") as f:
            json.dump({"images": images, "annotations": anns,
                       "categories": [{"id": c + 1, "name": CLASSES[c]} for c in range(num_classes)]}, f)
    return root


@pytest.fixture
def aitod_root(tmp_path):
    imgs, targets = synthetic_batch(4, 128, seed=3)
    return write_aitod(str(tmp_path / "aitodv2"), imgs, targets)
