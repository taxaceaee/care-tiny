"""The AI-TOD evaluator: exact equivalence with pycocotools' matching, per-category / per-block / streaming
evaluation equal to one joint pass, and image subsets."""

import json

import numpy as np
import pytest
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from care_tiny.aitod_eval import AITODeval, StreamingEvaluator, evaluate_aitod, summarize


def make_coco(path, n_img=25, seed=0):
    rng = np.random.RandomState(seed)
    images, anns, aid = [], [], 1
    for i in range(1, n_img + 1):
        images.append({"id": i, "file_name": f"{i}.png", "width": 100, "height": 100})
        for _ in range(rng.randint(0, 12)):
            s = float(rng.choice([3, 6, 10, 14, 20, 40]))
            w, h = s * rng.uniform(0.6, 1.4), s * rng.uniform(0.6, 1.4)
            x, y = rng.uniform(0, 100 - w), rng.uniform(0, 100 - h)
            anns.append({"id": aid, "image_id": i, "category_id": int(rng.randint(0, 3)), "bbox": [x, y, w, h],
                         "area": w * h, "iscrowd": 0})
            aid += 1
    with open(path, "w") as f:
        json.dump({"images": images, "annotations": anns,
                   "categories": [{"id": c, "name": f"c{c}"} for c in range(3)]}, f)
    return anns


def make_dets(anns, seed=0, perfect=False, n_img=25):
    rng = np.random.RandomState(seed)
    rows = []
    for a in anns:
        x, y, w, h = a["bbox"]
        if perfect:
            rows.append([a["image_id"], x, y, w, h, 0.9, a["category_id"]])
            continue
        for _ in range(rng.randint(0, 3)):
            j = rng.normal(0, 0.15 * max(w, h), 2)
            rows.append([a["image_id"], x + j[0], y + j[1], w * rng.uniform(0.8, 1.2), h * rng.uniform(0.8, 1.2),
                         rng.rand(), a["category_id"]])
        if rng.rand() < 0.3:  # exact duplicate: IoU ties
            rows.append([a["image_id"], x, y, w, h, 0.5, a["category_id"]])
    if not perfect:
        for i in range(1, n_img + 1):
            for _ in range(rng.randint(0, 6)):
                rows.append([i, rng.uniform(0, 90), rng.uniform(0, 90), rng.uniform(2, 30), rng.uniform(2, 30),
                             rng.rand(), rng.randint(0, 3)])
    return np.array(rows, dtype=np.float64).reshape(-1, 7)


class PycocotoolsEval(AITODeval):
    """AI-TOD parameters with pycocotools' own (slow) matching."""

    def evaluateImg(self, imgId, catId, aRng, maxDet):
        r = COCOeval.evaluateImg(self, imgId, catId, aRng, maxDet)
        if r is not None:
            r["dtIoU0"] = np.zeros(len(r["dtIds"]))
        return r


def _run(cls, gt, dets):
    E = cls(gt, gt.loadRes(dets), "bbox")
    E.evaluate()
    E.accumulate()
    return E


@pytest.fixture
def coco(tmp_path):
    p = str(tmp_path / "gt.json")
    return p, make_coco(p, seed=3)


def test_fast_matching_equals_pycocotools(tmp_path):
    p = str(tmp_path / "gt.json")
    anns = make_coco(p)
    gt = COCO(p)
    dets = make_dets(anns)
    fast, ref = _run(AITODeval, gt, dets), _run(PycocotoolsEval, gt, dets)
    for a, b in zip(fast.evalImgs, ref.evalImgs):
        if a is None or b is None:
            assert a is b
            continue
        for k in ("dtMatches", "gtMatches", "dtIgnore", "gtIgnore"):
            np.testing.assert_array_equal(np.asarray(a[k]), np.asarray(b[k]))
    np.testing.assert_array_equal(fast.eval["precision"], ref.eval["precision"])
    np.testing.assert_array_equal(fast.eval["recall"], ref.eval["recall"])


def test_per_category_blocks_equal_one_joint_pass(coco):
    p, anns = coco
    dets = make_dets(anns, seed=3)
    E = _run(AITODeval, COCO(p), dets)
    joint = summarize(E.eval["precision"], E.eval["recall"], E.accumulate_lrp())
    one = evaluate_aitod(p, dets, num_procs=1)
    for k, v in joint.items():
        assert one["stats"][k] == pytest.approx(v, abs=1e-12), k
    for ids in (None, sorted({a["image_id"] for a in anns})[::2]):
        ref = evaluate_aitod(p, dets, num_procs=1, img_ids=ids)
        for block_dets in (5, 40):
            blk = evaluate_aitod(p, dets, num_procs=1, img_ids=ids, block_dets=block_dets)
            assert blk["stats"] == ref["stats"] and blk["per_class"] == ref["per_class"]


def test_process_pool_equals_single_process(coco):
    p, anns = coco
    dets = make_dets(anns, seed=3)
    assert evaluate_aitod(p, dets, num_procs=3)["stats"] == evaluate_aitod(p, dets, num_procs=1)["stats"]


def test_perfect_detections(tmp_path):
    p = str(tmp_path / "gt.json")
    anns = make_coco(p, seed=5)
    s = evaluate_aitod(p, make_dets(anns, perfect=True), num_procs=1)["stats"]
    for k in ("AP", "AP50", "AP75", "APvt", "APt", "APs", "APm", "AR1500"):
        assert s[k] == pytest.approx(1.0, abs=1e-6), k
    assert s["oLRP"] == pytest.approx(0.0, abs=1e-6)


def test_streaming_equals_batch_evaluation(coco):
    """Blocks scored while inference runs (images fed in id order) give the evaluate_aitod result exactly, also with
    a category without detections and an image without any."""
    p, anns = coco
    dets = make_dets(anns, seed=3)
    dets = dets[dets[:, 6] != 2]  # category 2 never detected
    with open(p) as f:
        all_ids = sorted(im["id"] for im in json.load(f)["images"])
    dets = dets[dets[:, 0] != all_ids[1]]  # one image without detections
    for ids in (all_ids, all_ids[::2]):
        sel = dets[np.isin(dets[:, 0].astype(np.int64), ids)]
        ref = evaluate_aitod(p, sel, num_procs=1, img_ids=ids)
        for procs, n_blocks, bs in ((1, 3, 2), (1, 50, 5), (3, 0, 4)):
            ev = StreamingEvaluator(p, ids, num_procs=procs, n_blocks=n_blocks)
            for i in range(0, len(ids), bs):
                b_ids = ids[i:i + bs]
                ev.add(sel[np.isin(sel[:, 0].astype(np.int64), b_ids)], b_ids)
            res = ev.finish()
            assert res["stats"] == ref["stats"] and res["per_class"] == ref["per_class"]
    ev = StreamingEvaluator(p, all_ids, num_procs=1)
    with pytest.raises(ValueError):
        ev.add(dets[:0], all_ids[1:2])  # skips the first image


def test_subset_counts_only_evaluated_images(tmp_path):
    """GTs of images that were never run must not count as misses."""
    p = str(tmp_path / "gt.json")
    anns = make_coco(p, seed=5)
    keep = sorted({a["image_id"] for a in anns})[::3]
    dets = make_dets([a for a in anns if a["image_id"] in keep], perfect=True)
    s = evaluate_aitod(p, dets, num_procs=1, img_ids=keep)["stats"]
    for k in ("AP", "AP50", "AR1500"):
        assert s[k] == pytest.approx(1.0, abs=1e-6), k
    assert evaluate_aitod(p, dets, num_procs=1)["stats"]["AR1500"] < 0.6  # whole json: unseen GTs are misses


def test_results_table_lists_the_reference_rows():
    from care_tiny.aitod_eval import results_table
    stats = {k: 0.5 for k in ("AP", "AP50", "AP75", "APvt", "APt", "APs", "APm")}
    table = results_table(stats, {"name": "ours", "params": "10.0M"})
    assert "Dome-DETR-M (paper)" in table and "| ours | 10.0M | - | 50.0 |" in table
