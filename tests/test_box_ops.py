import torch
import torchvision

from care_tiny import box_ops

from conftest import rand_boxes


def test_iou_and_giou_match_torchvision():
    g = torch.Generator().manual_seed(0)
    a, b = rand_boxes(50, gen=g), rand_boxes(60, gen=g)
    assert torch.allclose(box_ops.iou_pairwise(a, b), torchvision.ops.box_iou(a, b), atol=1e-5)
    ref = torch.diag(torchvision.ops.generalized_box_iou(a, b[:50]))
    assert torch.allclose(box_ops.giou_aligned(a, b[:50]), ref, atol=1e-5)


def test_nwd_identity_and_symmetry():
    a, b = rand_boxes(20), rand_boxes(20)
    assert torch.allclose(box_ops.nwd_aligned(a, a, 12.8), torch.ones(20), atol=1e-3)
    assert torch.allclose(box_ops.nwd_aligned(a, b, 12.8), box_ops.nwd_aligned(b, a, 12.8))


def test_size_aware_metric_uses_nwd_for_small_boxes():
    """A 4 px box shifted by 4 px has IoU 0, but keeps a usable NWD."""
    gt = torch.tensor([[10.0, 10.0, 14.0, 14.0], [10.0, 10.0, 50.0, 50.0]])
    pred = gt + torch.tensor([4.0, 0.0, 4.0, 0.0])
    m = box_ops.size_aware_metric_aligned(pred, gt, 12.8, 16.0)
    assert float(box_ops.iou_aligned(pred[:1], gt[:1])) == 0.0
    assert 0.7 < float(m[0]) < 0.75
    assert torch.allclose(m[1], box_ops.iou_aligned(pred[1:], gt[1:])[0])
    pw = box_ops.size_aware_metric_pairwise(pred, gt, 12.8, 16.0)
    assert torch.allclose(torch.diag(pw), m)


def test_format_round_trip():
    b = rand_boxes(10)
    assert torch.allclose(box_ops.cxcywh_to_xyxy(box_ops.xyxy_to_cxcywh(b)), b, atol=1e-5)
    crossed = b[:, [2, 3, 0, 1]]
    assert torch.equal(box_ops.sort_xyxy(crossed), b)
