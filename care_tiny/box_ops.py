"""Box utilities: format conversions and aligned / pairwise IoU, GIoU and NWD."""

import torch

EPS = 1e-7


def cxcywh_to_xyxy(b):
    cx, cy, w, h = b.unbind(-1)
    return torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], -1)


def xyxy_to_cxcywh(b):
    x1, y1, x2, y2 = b.unbind(-1)
    return torch.stack([0.5 * (x1 + x2), 0.5 * (y1 + y2), x2 - x1, y2 - y1], -1)


def sort_xyxy(b):
    """Make x1 <= x2 and y1 <= y2 (refined edges may cross early in training)."""
    x1, y1, x2, y2 = b.unbind(-1)
    return torch.stack([torch.minimum(x1, x2), torch.minimum(y1, y2), torch.maximum(x1, x2), torch.maximum(y1, y2)], -1)


def _area(b):
    return (b[..., 2] - b[..., 0]).clamp(min=0) * (b[..., 3] - b[..., 1]).clamp(min=0)


def _wh_area(wh):
    """w * h of a (..., 2) tensor. Not wh.prod(-1): its backward checks the input for zeros with .item(), a host
    sync in every loss call."""
    return wh[..., 0] * wh[..., 1]


def iou_aligned(a, b):
    lt = torch.maximum(a[..., :2], b[..., :2])
    rb = torch.minimum(a[..., 2:], b[..., 2:])
    inter = _wh_area((rb - lt).clamp(min=0))
    union = _area(a) + _area(b) - inter
    return inter / (union + EPS)


def giou_aligned(a, b):
    lt = torch.maximum(a[..., :2], b[..., :2])
    rb = torch.minimum(a[..., 2:], b[..., 2:])
    inter = _wh_area((rb - lt).clamp(min=0))
    union = _area(a) + _area(b) - inter
    iou = inter / (union + EPS)
    elt = torch.minimum(a[..., :2], b[..., :2])
    erb = torch.maximum(a[..., 2:], b[..., 2:])
    encl = _wh_area((erb - elt).clamp(min=0))
    return iou - (encl - union) / (encl + EPS)


def iou_pairwise(a, b):
    """a (..., N, 4), b (..., M, 4) -> (..., N, M); leading dims (e.g. batch) are broadcast."""
    return iou_aligned(a[..., :, None, :], b[..., None, :, :])


def nwd_aligned(a, b, const):
    """Normalized Gaussian Wasserstein distance (Wang et al., 2021) of xyxy boxes."""
    ca, cb = xyxy_to_cxcywh(a), xyxy_to_cxcywh(b)
    d2 = (ca[..., 0] - cb[..., 0]) ** 2 + (ca[..., 1] - cb[..., 1]) ** 2
    d2 = d2 + 0.25 * ((ca[..., 2] - cb[..., 2]) ** 2 + (ca[..., 3] - cb[..., 3]) ** 2)
    return torch.exp(-torch.sqrt(d2.clamp(min=0) + EPS) / const)


def nwd_pairwise(a, b, const):
    return nwd_aligned(a[..., :, None, :], b[..., None, :, :], const)


def _gt_is_small(gt, small_thr):
    w = (gt[..., 2] - gt[..., 0]).clamp(min=0)
    h = (gt[..., 3] - gt[..., 1]).clamp(min=0)
    return torch.sqrt(w * h) < small_thr


def size_aware_metric_aligned(pred, gt, const, small_thr):
    """NWD for small GTs (sqrt(wh) < small_thr), IoU otherwise: IoU is too sensitive to a 1 px shift of a 4 px box."""
    return torch.where(_gt_is_small(gt, small_thr), nwd_aligned(pred, gt, const), iou_aligned(pred, gt))


def size_aware_metric_pairwise(pred, gt, const, small_thr):
    """pred (..., N, 4), gt (..., M, 4) -> (..., N, M)."""
    small = _gt_is_small(gt, small_thr)[..., None, :]
    return torch.where(small, nwd_pairwise(pred, gt, const), iou_pairwise(pred, gt))
