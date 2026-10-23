"""AI-TOD-v2 dataset in the official layout, augmentations, collate and DDP-safe samplers.

Official layout:
    <root>/annotations/aitodv2_{train,val,trainval,test}.json
    <root>/images/trainval/*.png, <root>/images/test/*.png
"""

import json
import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.v2.functional as TF
from torch.utils.data import Dataset, DistributedSampler, Sampler
from torchvision.io import ImageReadMode, decode_image, read_file

SPLITS = {
    "train": ("aitodv2_train.json", "trainval"),
    "val": ("aitodv2_val.json", "trainval"),
    "trainval": ("aitodv2_trainval.json", "trainval"),
    "test": ("aitodv2_test.json", "test"),
}
# copies of the dataset differ in folder naming (official: images/trainval; some mirrors: images/train)
IMG_DIR_CANDIDATES = {"trainval": ("trainval", "train", "val"), "test": ("test",)}
# the augmentations of the strong stage (data.aug.stage2_keep can keep some of them in the no-aug stage)
STRONG_OPS = frozenset({"copy_paste", "photometric", "scale", "rot90"})


def parse_stage2_keep(aug):
    keep = frozenset((aug or {}).get("stage2_keep", ()) or ())
    unknown = keep - STRONG_OPS
    if unknown:
        raise ValueError(f"data.aug.stage2_keep: unknown ops {sorted(unknown)} (known: {sorted(STRONG_OPS)})")
    return keep


def resolve_image_dirs(root, img_dir, file_names, probe=20):
    """Return the existing candidate image folders, best match first."""
    cands = [os.path.join(root, "images", c) for c in IMG_DIR_CANDIDATES[img_dir]]
    cands = [c for c in cands if os.path.isdir(c)]
    if not cands:
        raise FileNotFoundError(f"no image folder for {img_dir!r} under {root}/images")
    sample = file_names[:probe]
    cands.sort(key=lambda c: -sum(os.path.isfile(os.path.join(c, f)) for f in sample))
    return cands


class AITODv2(Dataset):
    """One split of AI-TOD-v2. Training samples go through copy-paste, photometric, scale-crop, flips and rot90.

    gpu_photometric: the worker only draws the photometric parameters (target["photo"]) and the trainer applies them
    on the GPU batch with photometric_batch(); target["valid"] is the real-content rectangle (x0, y0, x1, y1).
    """

    def __init__(self, root, split, train=False, aug=None, max_images=0, seed=0, gpu_photometric=False):
        ann_file, img_dir = SPLITS[split]
        self.root = root
        self.split = split
        self.ann_path = os.path.join(root, "annotations", ann_file)
        if not os.path.isfile(self.ann_path):
            raise FileNotFoundError(f"{self.ann_path} not found (set data.root to the AI-TOD-v2 folder)")
        with open(self.ann_path, encoding="utf-8") as f:
            d = json.load(f)
        cats = sorted(d["categories"], key=lambda c: c["id"])
        self.cat_ids = [c["id"] for c in cats]
        self.cat_names = [c["name"] for c in cats]
        self.cat2label = {c: i for i, c in enumerate(self.cat_ids)}
        per_img = {}
        for a in d["annotations"]:
            per_img.setdefault(a["image_id"], []).append(a)
        images = sorted(d["images"], key=lambda im: im["id"])
        if max_images and max_images < len(images):
            rng = random.Random(seed)
            images = sorted(rng.sample(images, max_images), key=lambda im: im["id"])
        self.images = []
        self.boxes = []  # xywh per image, float32
        self.labels = []
        for im in images:
            anns = per_img.get(im["id"], [])
            self.images.append((im["id"], im["file_name"], im["width"], im["height"]))
            self.boxes.append(np.array([a["bbox"] for a in anns], dtype=np.float32).reshape(-1, 4))
            self.labels.append(np.array([self.cat2label[a["category_id"]] for a in anns], dtype=np.int64))
        self.image_ids = [im[0] for im in self.images]
        self.img_dirs = resolve_image_dirs(root, img_dir, [im[1] for im in self.images])
        self.img_dir = self.img_dirs[0]
        self.train = train
        self.aug = aug or {}
        self.gpu_photometric = gpu_photometric
        self.strong = True  # strong stage; the trainer rebuilds the loader to switch (workers hold a copy)
        self.stage2_keep = parse_stage2_keep(self.aug)
        self.cp_bank = self._build_cp_bank() if train and self.aug.get("copy_paste_p", 0.0) > 0 else None

    def op_on(self, op):
        """A strong-stage op runs in the strong stage, and in the no-aug stage only if listed in stage2_keep."""
        return self.strong or op in self.stage2_keep

    def _build_cp_bank(self):
        """Instances of the rare classes for copy-paste: classes present in fewer than copy_paste_img_frac of the
        images. Returns {class: [(image index, box xyxy), ...]}."""
        a = self.aug
        n = max(1, len(self.labels))
        freq = np.zeros(len(self.cat_ids))
        for lab in self.labels:
            freq[np.unique(lab)] += 1
        rare = [c for c in range(len(self.cat_ids)) if 0 < freq[c] / n < a.get("copy_paste_img_frac", 0.05)]
        bank = {c: [] for c in rare}
        for i, (b, lab) in enumerate(zip(self.boxes, self.labels)):
            for (x, y, w, h), c in zip(b.tolist(), lab.tolist()):
                if c in bank and w >= 2 and h >= 2:
                    bank[c].append((i, (x, y, x + w, y + h)))
        return {c: v for c, v in bank.items() if v}

    def __len__(self):
        return len(self.images)

    def image_path(self, i):
        name = self.images[i][1]
        path = os.path.join(self.img_dir, name)
        if os.path.isfile(path):
            return path
        for d in self.img_dirs[1:]:
            p = os.path.join(d, name)
            if os.path.isfile(p):
                return p
        raise FileNotFoundError(f"{name} not found in {self.img_dirs}")

    def load_image(self, i):
        return decode_image(read_file(self.image_path(i)), mode=ImageReadMode.RGB)  # uint8 (3,H,W)

    def __getitem__(self, i):
        img_id, _, w0, h0 = self.images[i]
        img = self.load_image(i)
        xywh = torch.from_numpy(self.boxes[i])
        boxes = torch.cat([xywh[:, :2], xywh[:, :2] + xywh[:, 2:]], 1)
        labels = torch.from_numpy(self.labels[i])
        extra = {}
        if self.train:
            keep = (xywh[:, 2] > 0.01) & (xywh[:, 3] > 0.01)  # drop the handful of zero-size boxes
            boxes, labels = boxes[keep], labels[keep]
            if self.op_on("copy_paste") and self.cp_bank and random.random() < self.aug.get("copy_paste_p", 0.0):
                img, boxes, labels = self._copy_paste(img, boxes, labels)
            img, boxes, labels, photo, valid = self._augment(img, boxes, labels)
            extra = {"photo": photo, "valid": valid}
        target = {
            "boxes": boxes.float(),
            "labels": labels.long(),
            "image_id": int(img_id),
            # (h, w) of the input, for clipping the predictions
            "orig_size": (int(img.shape[1]), int(img.shape[2])) if self.train else (int(h0), int(w0)),
            **extra,
        }
        return img, target

    # ------------------------------------------------------------------ augmentation
    def _raw(self, j):
        img = self.load_image(j)
        xywh = torch.from_numpy(self.boxes[j])
        keep = (xywh[:, 2] > 0.01) & (xywh[:, 3] > 0.01)
        b = torch.cat([xywh[:, :2], xywh[:, :2] + xywh[:, 2:]], 1)[keep]
        return img, b, torch.from_numpy(self.labels[j])[keep]

    def _copy_paste(self, img, boxes, labels):
        """Paste rare-class instances (class drawn uniformly, then an instance of it; the other bank instances of the
        same source image come along) at random free places, at native size with a context margin of
        max(2 px, 25% of the size). The other source GTs inside a pasted crop keep their labels. A paste never covers
        an existing or an earlier pasted box (its margin rectangle must not intersect them), so no GT is hidden."""
        a = self.aug
        c = random.choice(list(self.cp_bank))
        j, _ = random.choice(self.cp_bank[c])
        src = self._raw(j)
        sb, sl = src[1], src[2]
        cand = [k for k in range(sb.shape[0]) if int(sl[k]) in self.cp_bank]
        random.shuffle(cand)
        cand = cand[: int(a.get("copy_paste_max", 8))]
        _, H, W = img.shape
        _, Hs, Ws = src[0].shape
        img = img.clone()
        occ = boxes.clone()
        new_b, new_l = [], []
        for k in cand:
            x1, y1, x2, y2 = sb[k].tolist()
            m = max(2.0, 0.25 * max(x2 - x1, y2 - y1))
            px1, py1 = int(max(0, math.floor(x1 - m))), int(max(0, math.floor(y1 - m)))
            px2, py2 = int(min(Ws, math.ceil(x2 + m))), int(min(Hs, math.ceil(y2 + m)))
            pw, ph = px2 - px1, py2 - py1
            if pw < 2 or ph < 2 or pw >= W or ph >= H:
                continue
            for _ in range(10):
                ox, oy = random.randint(0, W - pw), random.randint(0, H - ph)
                r = torch.tensor([ox, oy, ox + pw, oy + ph], dtype=occ.dtype)
                if occ.shape[0] == 0 or not bool(((occ[:, 0] < r[2]) & (occ[:, 2] > r[0]) &
                                                   (occ[:, 1] < r[3]) & (occ[:, 3] > r[1])).any()):
                    img[:, oy:oy + ph, ox:ox + pw] = src[0][:, py1:py2, px1:px2]
                    # every source GT inside the crop comes along; cut boxes follow the scale-crop visibility rule
                    sh = torch.tensor([ox - px1, oy - py1, ox - px1, oy - py1], dtype=sb.dtype)
                    cl = torch.stack([sb[:, 0].clamp(px1, px2), sb[:, 1].clamp(py1, py2),
                                      sb[:, 2].clamp(px1, px2), sb[:, 3].clamp(py1, py2)], 1)
                    cwh = cl[:, 2:] - cl[:, :2]
                    area0 = (sb[:, 2] - sb[:, 0]) * (sb[:, 3] - sb[:, 1])
                    inside = (cwh[:, 0] * cwh[:, 1] >= a.get("min_visible", 0.5) * area0) & \
                             (cwh[:, 0] >= a.get("min_size", 0.5)) & (cwh[:, 1] >= a.get("min_size", 0.5))
                    inside[k] = True  # the instance itself (always fully inside its margin)
                    new_b.append((cl[inside] + sh).to(boxes.dtype))
                    new_l.append(sl[inside])
                    occ = torch.cat([occ, r[None]])
                    break
        if not new_b:
            return img, boxes, labels
        return img, torch.cat([boxes] + new_b), torch.cat([labels] + new_l)

    def _augment(self, img, boxes, labels):
        """Returns img, boxes, labels, photo (ops still to apply on the GPU, [] if none) and the valid rectangle."""
        a = self.aug
        _, h, w = img.shape
        valid = torch.tensor([[0.0, 0.0, float(w), float(h)]])
        photo = []
        if self.op_on("photometric") and random.random() < a.get("photometric_p", 0.0):
            photo = photometric_params(a)
            if not self.gpu_photometric:
                img, photo = apply_photometric(img, photo), []
        if self.op_on("scale") and random.random() < a.get("scale_p", 0.0):
            img, boxes, labels, valid = self._scale_crop(img, boxes, labels)
        n = boxes.shape[0]
        allb = torch.cat([boxes, valid.to(boxes.dtype)])  # the valid rectangle follows the flips / rotations
        if random.random() < a.get("hflip_p", 0.0):
            img, allb = hflip(img, allb)
        if random.random() < a.get("vflip_p", 0.0):
            img, allb = vflip(img, allb)
        if self.op_on("rot90") and random.random() < a.get("rot90_p", 0.0):
            img, allb = rot90(img, allb, random.randint(1, 3))
        boxes = allb[:n]
        valid = tuple(int(round(v)) for v in allb[n].tolist())
        return img, boxes, labels, photo, valid

    def _scale_crop(self, img, boxes, labels):
        """Random rescale, then a random crop / zero pad back to the input size; boxes less than min_visible visible
        or thinner than min_size are dropped."""
        a = self.aug
        _, h, w = img.shape
        s = random.uniform(*a["scale_range"])
        nh, nw = max(1, round(h * s)), max(1, round(w * s))
        img = resize_uint8(img, nh, nw)
        sx, sy = nw / w, nh / h
        boxes = boxes * torch.tensor([sx, sy, sx, sy])
        ox = random.randint(0, nw - w) if nw > w else -random.randint(0, w - nw)
        oy = random.randint(0, nh - h) if nh > h else -random.randint(0, h - nh)
        canvas = torch.zeros((3, h, w), dtype=torch.uint8)
        sx0, sy0 = max(ox, 0), max(oy, 0)
        dx0, dy0 = max(-ox, 0), max(-oy, 0)
        cw, ch = min(nw - sx0, w - dx0), min(nh - sy0, h - dy0)
        canvas[:, dy0:dy0 + ch, dx0:dx0 + cw] = img[:, sy0:sy0 + ch, sx0:sx0 + cw]
        boxes = boxes - torch.tensor([ox, oy, ox, oy], dtype=boxes.dtype)
        clipped = torch.stack([boxes[:, 0].clamp(0, w), boxes[:, 1].clamp(0, h),
                               boxes[:, 2].clamp(0, w), boxes[:, 3].clamp(0, h)], 1)
        area0 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        cwh = clipped[:, 2:] - clipped[:, :2]
        area1 = cwh[:, 0] * cwh[:, 1]
        keep = (area1 >= a["min_visible"] * area0) & (cwh[:, 0] >= a["min_size"]) & (cwh[:, 1] >= a["min_size"])
        valid = torch.tensor([[float(dx0), float(dy0), float(dx0 + cw), float(dy0 + ch)]])
        return canvas, clipped[keep], labels[keep], valid


def resize_uint8(img, nh, nw):
    """Antialiased bilinear resize of a uint8 (3,H,W) image (native uint8 kernel, 4-5x faster than float32)."""
    return F.interpolate(img[None], size=(nh, nw), mode="bilinear", align_corners=False, antialias=True)[0]


# ----------------------------------------------------------------------------- photometric
PHOTO_OPS = (TF.adjust_brightness, TF.adjust_contrast, TF.adjust_saturation, TF.adjust_hue)


def photometric_params(a):
    """Random (op index, factor) list in a random order: brightness, contrast, saturation, hue."""
    ops = [(0, random.uniform(1 - a["brightness"], 1 + a["brightness"])),
           (1, random.uniform(1 - a["contrast"], 1 + a["contrast"])),
           (2, random.uniform(1 - a["saturation"], 1 + a["saturation"])),
           (3, random.uniform(-a["hue"], a["hue"]))]
    random.shuffle(ops)
    return ops


def apply_photometric(img, ops):
    """uint8 (3,H,W) on any device; the same torchvision ops on CPU and GPU."""
    for op, f in ops:
        img = PHOTO_OPS[op](img, f)
    return img


@torch.no_grad()
def photometric_batch(images, targets):
    """Apply each sample's photometric ops (drawn in the worker) in place to the uint8 batch on its device. Only the
    valid rectangle is processed, so crop borders and collate padding stay black and the contrast mean is taken over
    real pixels."""
    for i, t in enumerate(targets):
        ops = t.get("photo")
        if not ops:
            continue
        x0, y0, x1, y1 = t["valid"]
        if x1 <= x0 or y1 <= y0:
            continue
        images[i, :, y0:y1, x0:x1] = apply_photometric(images[i, :, y0:y1, x0:x1], ops)
    return images


def hflip(img, boxes):
    w = img.shape[-1]
    img = img.flip(-1)
    boxes = torch.stack([w - boxes[:, 2], boxes[:, 1], w - boxes[:, 0], boxes[:, 3]], 1)
    return img, boxes


def vflip(img, boxes):
    h = img.shape[-2]
    img = img.flip(-2)
    boxes = torch.stack([boxes[:, 0], h - boxes[:, 3], boxes[:, 2], h - boxes[:, 1]], 1)
    return img, boxes


def rot90(img, boxes, k):
    """Rotate by k * 90 degrees with torch.rot90(dims=(1, 2)); new(r, c) = old(c, W - 1 - r)."""
    for _ in range(k % 4):
        w = img.shape[-1]
        img = torch.rot90(img, 1, dims=(1, 2))
        boxes = torch.stack([boxes[:, 1], w - boxes[:, 2], boxes[:, 3], w - boxes[:, 0]], 1)
    return img, boxes


def collate(batch, pad_divisor=32, pad_to=0):
    """Zero-pad a batch to a multiple of pad_divisor. pad_to > 0 (training): every batch is exactly pad_to x pad_to,
    so a compiled model sees one image shape; larger images are cropped to pad_to (AI-TOD-v2 trainval holds one
    801 x 800 image) and their boxes clipped; boxes left under 1 px are dropped."""
    if pad_to:
        assert pad_to % pad_divisor == 0, (pad_to, pad_divisor)
        batch = [_crop_to(im, t, pad_to) if max(im.shape[1:]) > pad_to else (im, t) for im, t in batch]
    imgs, targets = zip(*batch)
    hmax = max(max(i.shape[1] for i in imgs), pad_to)
    wmax = max(max(i.shape[2] for i in imgs), pad_to)
    hp = int(math.ceil(hmax / pad_divisor) * pad_divisor)
    wp = int(math.ceil(wmax / pad_divisor) * pad_divisor)
    out = torch.zeros((len(imgs), 3, hp, wp), dtype=torch.uint8)
    for i, im in enumerate(imgs):
        out[i, :, : im.shape[1], : im.shape[2]] = im
    return out, list(targets)


def _crop_to(img, t, size):
    img = img[:, :size, :size]
    b = t["boxes"].clone()
    b[:, 0::2] = b[:, 0::2].clamp(0, img.shape[2])
    b[:, 1::2] = b[:, 1::2].clamp(0, img.shape[1])
    keep = ((b[:, 2] - b[:, 0]) >= 1) & ((b[:, 3] - b[:, 1]) >= 1)
    t = {**t, "boxes": b[keep], "labels": t["labels"][keep], "orig_size": (int(img.shape[1]), int(img.shape[2]))}
    if "valid" in t:
        x0, y0, x1, y1 = t["valid"]
        t["valid"] = (min(x0, img.shape[2]), min(y0, img.shape[1]), min(x1, img.shape[2]), min(y1, img.shape[1]))
    return img, t


# ----------------------------------------------------------------------------- samplers
class ShardSampler(Sampler):
    """Deterministic, non-padded sharding for evaluation (no image is scored twice)."""

    def __init__(self, n, rank, world):
        self.idx = list(range(rank, n, world))

    def __iter__(self):
        return iter(self.idx)

    def __len__(self):
        return len(self.idx)


def repeat_factors(ds, thr):
    """LVIS repeat factors r_i = max over the classes c of image i of max(1, sqrt(thr / f_c)), f_c = image frequency.

    AI-TOD-v2 AP is the mean over 8 classes, but wind-mill / airplane / swimming-pool / bridge appear in 0.2-2% of the
    trainval images (vehicle: 87% of the boxes); uniform sampling shows them ~100x less than the common classes.
    """
    n = len(ds.labels)
    img_freq = np.zeros(len(ds.cat_ids))
    for lab in ds.labels:
        img_freq[np.unique(lab)] += 1
    rc = np.maximum(1.0, np.sqrt(thr / np.maximum(img_freq / max(1, n), 1e-12)))
    return np.array([rc[np.unique(lab)].max() if lab.size else 1.0 for lab in ds.labels])


class RepeatFactorSampler(Sampler):
    """Epoch of fixed length round(sum r_i): image i appears floor(r_i) times plus once more with probability
    frac(r_i) (stochastic rounding seeded by the epoch), shuffled and sharded over the ranks. ``skip`` drops the
    first samples (mid-epoch resume)."""

    def __init__(self, factors, num_replicas, rank, seed):
        self.r = np.asarray(factors, dtype=np.float64)
        self.world, self.rank, self.seed, self.epoch, self.skip = num_replicas, rank, seed, 0, 0
        self.total = int(round(self.r.sum())) // num_replicas * num_replicas
        self.num_samples = self.total // num_replicas

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _indices(self):
        rng = np.random.default_rng(self.seed + 7919 * self.epoch)
        reps = np.floor(self.r).astype(np.int64) + (rng.random(self.r.size) < self.r - np.floor(self.r))
        idx = np.repeat(np.arange(self.r.size), reps)
        rng.shuffle(idx)
        if idx.size < self.total:  # stochastic rounding came out short: top up with random images
            idx = np.concatenate([idx, rng.integers(0, self.r.size, self.total - idx.size)])
        return idx[: self.total]

    def __iter__(self):
        return iter(self._indices()[self.rank::self.world][self.skip:].tolist())

    def __len__(self):
        return max(0, self.num_samples - self.skip)


class EpochSampler(DistributedSampler):
    """Uniform training sampler (DistributedSampler order, also for one process) that can start an epoch at a sample
    offset, so a persistent loader is reused across epochs and a mid-epoch resume skips the consumed batches without
    decoding them."""

    def __init__(self, dataset, num_replicas, rank, seed):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=True, seed=seed, drop_last=True)
        self.skip = 0

    def __iter__(self):
        return iter(list(super().__iter__())[self.skip:])

    def __len__(self):
        return max(0, self.num_samples - self.skip)
