"""AI-TOD evaluation: cocoapi-aitod parameters + oLRP, evaluated exactly per category.

cocoapi-aitod (jwwangchn/cocoapi-aitod) uses
    maxDets = [1, 100, 1500]
    areaRng = all [0, 1e5^2], verytiny [0, 8^2], tiny [8^2, 16^2], small [16^2, 32^2], medium [32^2, 1e5^2]
    oLRP at IoU 0.5, area all, maxDets 1500:
        lrp = (sum_TP (1-IoU)/(1-0.5) + FP + FN) / (TP + FP + FN), minimised over score thresholds.
COCOeval treats every category independently (IoU, matching and accumulation), so evaluating
one category at a time and concatenating the per-category arrays is identical to a joint run
while keeping memory bounded and allowing parallel workers.
"""

import atexit
import contextlib
import copy
import io
import multiprocessing as mp
import os

import numpy as np
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

AREA_RNG = [[0.0, 1e10], [0.0, 8.0 ** 2], [8.0 ** 2, 16.0 ** 2], [16.0 ** 2, 32.0 ** 2], [32.0 ** 2, 1e10]]
AREA_LBL = ["all", "verytiny", "tiny", "small", "medium"]
MAX_DETS = [1, 100, 1500]
IOU_THRS = np.linspace(0.5, 0.95, 10)


class AITODeval(COCOeval):
    def __init__(self, cocoGt=None, cocoDt=None, iouType="bbox"):
        super().__init__(cocoGt, cocoDt, iouType)
        self.params.maxDets = list(MAX_DETS)
        self.params.areaRng = [list(a) for a in AREA_RNG]
        self.params.areaRngLbl = list(AREA_LBL)

    def evaluateImg(self, imgId, catId, aRng, maxDet):
        """Exact re-implementation of pycocotools' greedy matching, vectorised over IoU thresholds.

        pycocotools loops T x D x G in Python (seconds per crowded AI-TOD image). Here each detection
        is processed once with numpy over (T, G); detections whose best IoU is below the smallest
        threshold can never match and are skipped. Tie rule kept: the *last* GT reaching the max IoU
        wins, non-ignored GTs are preferred over ignored ones, crowd GTs can be re-matched.
        Also records the IoU of the IoU=0.5 match (dtIoU0) for oLRP.
        """
        p = self.params
        gt = self._gts[imgId, catId]
        dt = self._dts[imgId, catId]
        if len(gt) == 0 and len(dt) == 0:
            return None
        for g in gt:
            g["_ignore"] = 1 if (g["ignore"] or g["area"] < aRng[0] or g["area"] > aRng[1]) else 0
        gtind = np.argsort([g["_ignore"] for g in gt], kind="mergesort")
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d["score"] for d in dt], kind="mergesort")
        dt = [dt[i] for i in dtind[0:maxDet]]
        iscrowd = np.array([int(o["iscrowd"]) for o in gt], dtype=bool)
        raw_ious = self.ious[imgId, catId]
        T, G, D = len(p.iouThrs), len(gt), len(dt)
        gtm = np.zeros((T, G))
        dtm = np.zeros((T, D))
        gtIg = np.array([g["_ignore"] for g in gt])
        dtIg = np.zeros((T, D))
        iou0 = np.zeros(D)
        if len(raw_ious) and G and D:
            ious = np.asarray(raw_ious)[:, gtind]
            thr_1d = np.minimum(np.asarray(p.iouThrs), 1 - 1e-10)
            thr = thr_1d[:, None]
            gid = np.array([g["id"] for g in gt], dtype=np.float64)
            ign = gtIg.astype(bool)
            matched = np.zeros((T, G), dtype=bool)
            mask0 = ious >= float(thr_1d.min())
            counts = mask0.sum(1)
            cand_d = np.nonzero(counts > 0)[0]
            rev = np.arange(G - 1, -1, -1)
            for d in cand_d:
                if counts[d] == 1:
                    g = int(mask0[d].argmax())
                    v = float(ious[d, g])
                    ok = (v >= thr_1d) & ((~matched[:, g]) | iscrowd[g])
                    if ok.any():
                        dtIg[ok, d] = gtIg[g]
                        dtm[ok, d] = gid[g]
                        gtm[ok, g] = dt[d]["id"]
                        matched[ok, g] = True
                        if ok[0]:
                            iou0[d] = v
                    continue
                row = ious[d][None, :]
                cand = ((~matched) | iscrowd[None, :]) & (row >= thr)
                m = np.full(T, -1)
                for part in (~ign, ign):
                    c = cand & part[None, :]
                    need = (m < 0) & c.any(1)
                    if not need.any():
                        continue
                    val = np.where(c, row, -np.inf)
                    mx = val.max(1, keepdims=True)
                    last = G - 1 - np.argmax((val == mx)[:, rev], axis=1)
                    m = np.where(need, last, m)
                ok = m >= 0
                if ok.any():
                    tt = np.nonzero(ok)[0]
                    mm = m[ok]
                    dtIg[tt, d] = gtIg[mm]
                    dtm[tt, d] = gid[mm]
                    gtm[tt, mm] = dt[d]["id"]
                    matched[tt, mm] = True
                    if ok[0]:
                        iou0[d] = ious[d, m[0]]
        a = np.array([d["area"] < aRng[0] or d["area"] > aRng[1] for d in dt]).reshape((1, D))
        dtIg = np.logical_or(dtIg, np.logical_and(dtm == 0, np.repeat(a, T, 0)))
        return {"image_id": imgId, "category_id": catId, "aRng": aRng, "maxDet": maxDet,
                "dtIds": [d["id"] for d in dt], "gtIds": [g["id"] for g in gt], "dtMatches": dtm,
                "gtMatches": gtm, "dtScores": [d["score"] for d in dt], "gtIgnore": gtIg, "dtIgnore": dtIg,
                "dtIoU0": iou0}

    def accumulate_lrp(self):
        """Per-category oLRP and components at IoU 0.5, area all, maxDets 1500."""
        p = self.params
        K, A, n_img = len(p.catIds), len(p.areaRng), len(p.imgIds)
        md = p.maxDets[-1]
        out = -np.ones((K, 4))  # olrp, loc, fp, fn
        for k in range(K):
            E = [self.evalImgs[k * A * n_img + i] for i in range(n_img)]
            E = [e for e in E if e is not None]
            if not E:
                continue
            npig = int(sum(np.count_nonzero(e["gtIgnore"] == 0) for e in E))
            if npig == 0:
                continue
            scores = np.concatenate([e["dtScores"][:md] for e in E])
            inds = np.argsort(-scores, kind="mergesort")
            dtm = np.concatenate([e["dtMatches"][0, :md] for e in E])[inds]
            dtig = np.concatenate([e["dtIgnore"][0, :md] for e in E])[inds].astype(bool)
            iou = np.concatenate([e["dtIoU0"][:md] for e in E])[inds]
            tps = (dtm != 0) & ~dtig
            fps = (dtm == 0) & ~dtig
            tp = np.cumsum(tps).astype(np.float64)
            fp = np.cumsum(fps).astype(np.float64)
            fn = npig - tp
            if tp.size == 0 or tp[-1] == 0:
                out[k] = [1.0, -1.0, -1.0, 1.0]
                continue
            loc = np.cumsum(np.where(tps, 1.0 - iou, 0.0))
            lrp = (loc / (1 - p.iouThrs[0]) + fp + fn) / (tp + fp + fn)
            j = int(np.argmin(lrp))
            out[k] = [lrp[j], loc[j] / tp[j] if tp[j] > 0 else -1.0,
                      fp[j] / (tp[j] + fp[j]) if tp[j] + fp[j] > 0 else -1.0, fn[j] / npig]
        return out


# ----------------------------------------------------------------------------- per-category workers & caches
_G = {}
_GT_META_CACHE = {}
_POOL_STATE = {"key": None, "pool": None}


def _get_gt_meta(gt_path):
    """Cache parsed COCO ground-truth metadata in the main process keyed by (abspath, mtime)."""
    ap = os.path.abspath(gt_path)
    key = (ap, os.path.getmtime(ap))
    cached = _GT_META_CACHE.get(key)
    if cached is not None:
        return cached
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(gt_path)
    cats = sorted(gt.dataset["categories"], key=lambda c: c["id"])
    all_img_ids = np.array(sorted({int(im["id"]) for im in gt.dataset["images"]}), dtype=np.int64)
    _GT_META_CACHE.clear()
    _GT_META_CACHE[key] = (gt, cats, all_img_ids)
    return gt, cats, all_img_ids


def _shutdown_pool(terminate=False):
    pool = _POOL_STATE.get("pool")
    _POOL_STATE["key"] = None
    _POOL_STATE["pool"] = None
    if pool is not None:
        try:
            if terminate:
                pool.terminate()
            else:
                pool.close()
            pool.join()
        except Exception:
            pass


atexit.register(_shutdown_pool)


def _get_pool(gt_path, num_procs):
    """Reuse a persistent spawn worker pool across epochs when evaluating the same GT file with num_procs workers."""
    ap = os.path.abspath(gt_path)
    key = (ap, os.path.getmtime(ap), int(num_procs))
    if _POOL_STATE["pool"] is not None and _POOL_STATE["key"] == key:
        return _POOL_STATE["pool"]
    _shutdown_pool()
    pool = mp.get_context("spawn").Pool(int(num_procs), initializer=_init_worker, initargs=(gt_path,))
    _POOL_STATE["key"] = key
    _POOL_STATE["pool"] = pool
    return pool


def _init_worker(gt_path, img_ids=None):
    ap = os.path.abspath(gt_path)
    mtime = os.path.getmtime(ap)
    if _G.get("gt_key") != (ap, mtime) or "gt" not in _G:
        with contextlib.redirect_stdout(io.StringIO()):
            _G["gt"] = COCO(gt_path)
        _G["gt_key"] = (ap, mtime)
    # Only the images that were actually run count. With max_*_images subsets the json still holds every image of
    # the split: evaluating all of them turned the GT of the skipped images into false negatives (500 of 2804 val
    # images capped recall, AP and AR at ~18% of their true value).
    _G["img_ids"] = sorted(img_ids) if img_ids is not None else sorted(_G["gt"].getImgIds())


def _get_cat_gt_index(gt, cat_id):
    """Lazily pre-pack per-image GT numpy arrays for `cat_id` on the COCO instance `gt`."""
    cache = getattr(gt, "_aitod_cat_idx", None)
    if cache is None:
        cache = {}
        gt._aitod_cat_idx = cache
    cid = int(cat_id)
    cat_map = cache.get(cid)
    if cat_map is not None:
        return cat_map
    cat_map = {}
    for img_id, anns in gt.imgToAnns.items():
        g = [a for a in anns if a["category_id"] == cid]
        if g:
            boxes = np.ascontiguousarray([a["bbox"] for a in g], dtype=np.float64)
            areas = np.ascontiguousarray([a["area"] for a in g], dtype=np.float64)
            isc_u8 = np.ascontiguousarray([int(a.get("iscrowd", 0)) for a in g], dtype=np.uint8)
            base_ig = np.ascontiguousarray([bool(a.get("iscrowd", 0) or a.get("ignore", 0)) for a in g], dtype=bool)
            cat_map[int(img_id)] = (boxes, areas, isc_u8, isc_u8.astype(bool), base_ig)
    cache[cid] = cat_map
    return cat_map


def _empty_result(gt, cat_id, img_ids=None):
    ids = img_ids if img_ids is not None else _G["img_ids"]
    ann_ids = gt.getAnnIds(imgIds=ids, catIds=[cat_id], iscrowd=False)
    areas = np.array([a["area"] for a in gt.loadAnns(ann_ids)])
    T, R = len(IOU_THRS), 101
    prec = -np.ones((T, R, 1, len(AREA_RNG), len(MAX_DETS)))
    rec = -np.ones((T, 1, len(AREA_RNG), len(MAX_DETS)))
    for a, (lo, hi) in enumerate(AREA_RNG):
        if areas.size and np.any((areas >= lo) & (areas <= hi)):
            prec[:, :, :, a, :] = 0
            rec[:, :, a, :] = 0
    lrp = np.array([[1.0, -1.0, -1.0, 1.0]]) if areas.size else -np.ones((1, 4))
    return prec, rec, lrp


def _eval_chunk(args):
    """Per-image matching (COCOeval.evaluate) of one category on one contiguous block of the sorted image ids.

    Operates directly on the (N, 7) numpy array without constructing Python dicts in COCO.loadRes or COCOeval._prepare,
    sorts detections once per image instead of 6 times, and caches matching across area ranges that share the same
    GT ignore mask. Produces bit-for-bit identical outputs to AITODeval.evaluate()."""
    cat_id, part, imgs, dets = args  # a block without detections still runs: its GT are misses
    gt = _G["gt"]
    gt_map = _get_cat_gt_index(gt, cat_id)
    imgs_list = [int(i) for i in imgs]
    imgs_arr = np.asarray(imgs_list, dtype=np.int64)
    n_imgs = len(imgs_list)
    A = len(AREA_RNG)
    T = len(IOU_THRS)
    max_det = MAX_DETS[-1]
    thr_1d = np.minimum(IOU_THRS, 1.0 - 1e-10)
    thr_col = thr_1d[:, None]
    thr_min = float(thr_1d[0])

    if dets.shape[0] > 0:
        if not np.all(dets[:, 6] == cat_id):
            dets = dets[dets[:, 6] == cat_id]
    if dets.shape[0] > 0:
        img_col = dets[:, 0].astype(np.int64)
        if dets.shape[0] > 1 and not np.all(img_col[:-1] <= img_col[1:]):
            order = np.argsort(img_col, kind="mergesort")
            dets = dets[order]
            img_col = img_col[order]
        starts = np.searchsorted(img_col, imgs_arr, side="left")
        ends = np.searchsorted(img_col, imgs_arr, side="right")
    else:
        starts = np.zeros(n_imgs, dtype=np.int64)
        ends = np.zeros(n_imgs, dtype=np.int64)

    slim = [None] * (A * n_imgs)
    gt_ig_empty = np.empty(0, dtype=np.int64)

    for idx_i, img_id in enumerate(imgs_list):
        s, e = int(starts[idx_i]), int(ends[idx_i])
        gt_entry = gt_map.get(img_id)
        D_raw = e - s
        if gt_entry is None and D_raw == 0:
            continue

        if D_raw > 0:
            d_slice = dets[s:e]
            dtind = np.argsort(-d_slice[:, 5], kind="mergesort")[:max_det]
            d_sorted = d_slice[dtind]
            dt_scores = np.ascontiguousarray(d_sorted[:, 5], dtype=np.float64)
            dt_boxes = np.ascontiguousarray(d_sorted[:, 1:5], dtype=np.float64)
            dt_areas = dt_boxes[:, 2] * dt_boxes[:, 3]
            D = dt_scores.shape[0]
        else:
            dt_scores = np.empty(0, dtype=np.float64)
            dt_areas = np.empty(0, dtype=np.float64)
            D = 0

        if gt_entry is None:
            dtm_empty = np.zeros((T, D), dtype=bool)
            iou0_empty = np.zeros(D, dtype=np.float64)
            for a_idx, (lo, hi) in enumerate(AREA_RNG):
                da = (dt_areas < lo) | (dt_areas > hi)
                slim[a_idx * n_imgs + idx_i] = {
                    "dtScores": dt_scores,
                    "dtMatches": dtm_empty,
                    "dtIgnore": np.broadcast_to(da[None, :], (T, D)).copy(),
                    "gtIgnore": gt_ig_empty,
                    "dtIoU0": iou0_empty,
                }
            continue

        gt_boxes, gt_areas, iscrowd_u8, iscrowd_b, gt_base_ig = gt_entry
        G = gt_areas.shape[0]
        if D == 0:
            dtm_empty = np.zeros((T, 0), dtype=bool)
            dtig_empty = np.zeros((T, 0), dtype=bool)
            iou0_empty = np.zeros(0, dtype=np.float64)
            for a_idx, (lo, hi) in enumerate(AREA_RNG):
                gt_ig0 = (gt_base_ig | (gt_areas < lo) | (gt_areas > hi)).astype(np.int64)
                gtind = np.argsort(gt_ig0, kind="mergesort")
                slim[a_idx * n_imgs + idx_i] = {
                    "dtScores": dt_scores,
                    "dtMatches": dtm_empty,
                    "dtIgnore": dtig_empty,
                    "gtIgnore": gt_ig0[gtind],
                    "dtIoU0": iou0_empty,
                }
            continue

        raw_ious = np.asarray(maskUtils.iou(dt_boxes, gt_boxes, iscrowd_u8), dtype=np.float64)
        mask0 = raw_ious >= thr_min
        counts = mask0.sum(1)
        cand_d = np.nonzero(counts > 0)[0]
        if cand_d.size:
            cand_cnt_list = counts[cand_d].tolist()
            cand_g0 = mask0[cand_d].argmax(1)
            cand_v0 = raw_ious[cand_d, cand_g0]
            cand_ge_thr = cand_v0[:, None] >= thr_1d[None, :]
            cand_g0_list = cand_g0.tolist()
            cand_v0_list = cand_v0.tolist()
            cand_d_list = cand_d.tolist()
            multi_subs = {int(d): np.nonzero(mask0[d])[0] for d, c in zip(cand_d_list, cand_cnt_list) if c > 1}
        else:
            cand_d_list = []

        cache = {}
        for a_idx, (lo, hi) in enumerate(AREA_RNG):
            gt_ig_bool = gt_base_ig | (gt_areas < lo) | (gt_areas > hi)
            key = gt_ig_bool.tobytes()
            cached = cache.get(key)
            if cached is None:
                gt_ig0 = gt_ig_bool.astype(np.int64)
                gtind = np.argsort(gt_ig0, kind="mergesort")
                gtIg = gt_ig0[gtind]
                dtm_T = np.zeros((D, T), dtype=bool)
                dtIg_T = np.zeros((D, T), dtype=bool)
                iou0 = np.zeros(D, dtype=np.float64)
                if cand_d_list:
                    matched_raw = np.zeros((G, T), dtype=bool)
                    ig_list = gt_ig_bool.tolist()
                    isc_list = iscrowd_b.tolist()
                    inv_gtind = np.empty(G, dtype=np.int64)
                    inv_gtind[gtind] = np.arange(G, dtype=np.int64)
                    for idx_c, d in enumerate(cand_d_list):
                        if cand_cnt_list[idx_c] == 1:
                            g = cand_g0_list[idx_c]
                            ge = cand_ge_thr[idx_c]
                            ok = ge if isc_list[g] else (ge & (~matched_raw[g]))
                            if ok.any():
                                matched_raw[g] |= ok
                                dtm_T[d] = ok
                                if ig_list[g]:
                                    dtIg_T[d] = ok
                                if ok[0]:
                                    iou0[d] = cand_v0_list[idx_c]
                        else:
                            sub_raw = multi_subs[d]
                            sub_ord = sub_raw[np.argsort(inv_gtind[sub_raw], kind="mergesort")]
                            k = sub_ord.size
                            row = raw_ious[d, sub_ord][None, :]
                            isc_s = iscrowd_b[sub_ord]
                            ign_s = gt_ig_bool[sub_ord]
                            cand = ((~matched_raw[sub_ord].T) | isc_s[None, :]) & (row >= thr_col)
                            if not cand.any():
                                continue
                            m = np.full(T, -1, dtype=np.int64)
                            rev = np.arange(k - 1, -1, -1)
                            for part_mask in (~ign_s, ign_s):
                                if not part_mask.any():
                                    continue
                                c = cand & part_mask[None, :]
                                need = (m < 0) & c.any(1)
                                if not need.any():
                                    continue
                                val = np.where(c, row, -np.inf)
                                mx = val.max(1, keepdims=True)
                                last = k - 1 - np.argmax((val == mx)[:, rev], axis=1)
                                m = np.where(need, sub_ord[last], m)
                            ok = m >= 0
                            if ok.any():
                                tt = np.nonzero(ok)[0]
                                mm = m[ok]
                                dtIg_T[d, tt] = gt_ig_bool[mm]
                                dtm_T[d, tt] = True
                                matched_raw[mm, tt] = True
                                if ok[0]:
                                    iou0[d] = raw_ious[d, m[0]]
                cached = (gtIg, np.ascontiguousarray(dtm_T.T), np.ascontiguousarray(dtIg_T.T), iou0)
                cache[key] = cached
            gtIg, dtm, dtIg_match, iou0 = cached
            da = (dt_areas < lo) | (dt_areas > hi)
            dtIg = dtIg_match | ((~dtm) & da[None, :])
            slim[a_idx * n_imgs + idx_i] = {
                "dtScores": dt_scores,
                "dtMatches": dtm,
                "dtIgnore": dtIg,
                "gtIgnore": gtIg,
                "dtIoU0": iou0,
            }
    return cat_id, part, imgs, slim


def _accumulate_category(cat_id, parts, img_ids):
    """Stitch the per-block evalImgs back into COCOeval order (area-major, then image) and accumulate."""
    A = len(AREA_RNG)
    parts = sorted(parts, key=lambda p: p[0])
    ev = []
    for a in range(A):
        for _, imgs, slim in parts:
            ev.extend(slim[a * len(imgs):(a + 1) * len(imgs)])
    E = AITODeval(None, None, "bbox")
    E.params.catIds = [cat_id]
    E.params.imgIds = list(img_ids)
    E._paramsEval = copy.deepcopy(E.params)
    E.evalImgs = ev
    with contextlib.redirect_stdout(io.StringIO()):
        E.accumulate()
        lrp = E.accumulate_lrp()
    return E.eval["precision"], E.eval["recall"], lrp


def _split_blocks(img_ids, dets, n_blocks):
    """Contiguous blocks of the sorted image ids holding about the same number of detections each."""
    if n_blocks <= 1 or len(img_ids) <= 1:
        return [img_ids]
    pos = np.searchsorted(img_ids, dets[:, 0].astype(np.int64))
    cum = np.cumsum(np.bincount(pos, minlength=len(img_ids)))
    cuts = np.unique(np.searchsorted(cum, np.arange(1, n_blocks) * cum[-1] / n_blocks, side="right"))
    bounds = [0] + [int(c) for c in cuts if 0 < c < len(img_ids)] + [len(img_ids)]
    return [img_ids[s:e] for s, e in zip(bounds[:-1], bounds[1:]) if e > s]


def _in_sorted_ids(img_col, sorted_ids):
    """Fast exact membership test of int64 `img_col` in sorted unique int64 `sorted_ids` (replaces np.isin)."""
    if sorted_ids.size == 0 or img_col.size == 0:
        return np.zeros(img_col.shape[0], dtype=bool)
    lo, hi = int(sorted_ids[0]), int(sorted_ids[-1])
    if hi - lo + 1 == sorted_ids.size:
        return (img_col >= lo) & (img_col <= hi)
    pos = np.searchsorted(sorted_ids, np.clip(img_col, lo, hi))
    return sorted_ids[pos] == img_col


def evaluate_aitod(gt_path, dets, num_procs=4, img_ids=None, block_dets=None):
    """dets: float array (N,7) [image_id, x, y, w, h, score, category_id] with official category ids.
    img_ids: the evaluated images (None = every image of the json). Pass it whenever only a subset was run.
    block_dets: detections per work item (default: N / (2 * num_procs)). Work used to be split by category only, so
    vehicle (~87% of the AI-TOD boxes) ran in a single process while the other workers idled; the matching is now
    split over image blocks too and re-assembled in COCOeval order (identical results, see unit test).

    Returns {"stats": {...}, "per_class": {...}}.
    """
    gt_main, cats, default_ids = _get_gt_meta(gt_path)
    cat_ids = [c["id"] for c in cats]
    all_ids = np.array(sorted({int(i) for i in img_ids}), dtype=np.int64) if img_ids is not None else default_ids
    dets = np.asarray(dets, dtype=np.float64).reshape(-1, 7)
    if dets.shape[0] > 0 and img_ids is not None:
        dets = dets[_in_sorted_ids(dets[:, 0].astype(np.int64), all_ids)]
    if dets.shape[0] > 1:
        img_col_all = dets[:, 0].astype(np.int64)
        if not np.all(img_col_all[:-1] <= img_col_all[1:]):
            dets = dets[np.argsort(img_col_all, kind="mergesort")]
    if block_dets is None:
        block_dets = dets.shape[0] / (2 * num_procs) if num_procs > 1 else float("inf")
    tasks, empty = [], []
    for c in cat_ids:
        dc = dets[dets[:, 6] == c]
        if dc.shape[0] == 0:  # no detection of this category at all
            empty.append(c)
            continue
        dc_imgs = dc[:, 0].astype(np.int64)
        n_blocks = int(np.ceil(dc.shape[0] / max(1.0, block_dets))) if np.isfinite(block_dets) else 1
        for part, imgs in enumerate(_split_blocks(all_ids, dc, n_blocks)):
            s = int(np.searchsorted(dc_imgs, int(imgs[0]), side="left"))
            e = int(np.searchsorted(dc_imgs, int(imgs[-1]), side="right"))
            tasks.append((c, part, imgs, dc[s:e]))
    tasks.sort(key=lambda t: -t[3].shape[0])  # biggest blocks first
    if num_procs > 1 and tasks:
        pool = _get_pool(gt_path, num_procs)
        chunks = list(pool.imap_unordered(_eval_chunk, tasks, chunksize=1))
    else:
        _G["gt"] = gt_main
        _G["img_ids"] = all_ids.tolist()
        chunks = [_eval_chunk(t) for t in tasks]
    by_cat = {}
    for c, part, imgs, slim in chunks:
        by_cat.setdefault(c, []).append((part, imgs, slim))
    all_ids_list = all_ids.tolist()
    results = {c: _accumulate_category(c, parts, all_ids_list) for c, parts in by_cat.items()}
    for c in empty:
        results[c] = _empty_result(gt_main, c, img_ids=all_ids_list)
    prec = np.concatenate([results[c][0] for c in cat_ids], axis=2)
    rec = np.concatenate([results[c][1] for c in cat_ids], axis=1)
    lrp = np.concatenate([results[c][2] for c in cat_ids], axis=0)
    stats = summarize(prec, rec, lrp)
    per_class = {}
    for k, c in enumerate(cats):
        s = prec[:, :, k, 0, -1]
        v = s[s > -1]
        per_class[c["name"]] = {"AP": float(v.mean()) if v.size else -1.0, "oLRP": float(lrp[k, 0])}
    return {"stats": stats, "per_class": per_class}


class StreamingEvaluator:
    """evaluate_aitod over detections that arrive in image order, scored while inference goes on.

    Single-GPU inference walks the dataset in its (sorted) image-id order, so the image ids are cut into n_blocks
    contiguous blocks and the per-category matching of a block (_eval_chunk, one task per category and block) is
    sent to the worker pool as soon as inference has passed the block. The pool is persistent across epochs so
    workers parse the GT json and build the category index only once. finish() stitches the blocks in COCOeval
    order exactly as evaluate_aitod does (any contiguous split gives the same result, see
    test_image_blocks_equal_single_pass), and a category without any detection gets _empty_result as there, so
    the output equals evaluate_aitod's.

        ev = StreamingEvaluator(gt_path, ds.image_ids, num_procs)
        for each batch: ev.add(dets_of_batch, image_ids_of_batch)   # ids in the order of ds.image_ids
        result = ev.finish()
    """

    def __init__(self, gt_path, img_ids, num_procs=4, n_blocks=0):
        self.gt_main, self.cats, _ = _get_gt_meta(gt_path)
        self.gt_path = gt_path
        self.order = [int(i) for i in img_ids]  # inference order
        self.ids = np.array(sorted(set(self.order)), dtype=np.int64)
        if self.order != self.ids.tolist():
            raise ValueError("StreamingEvaluator needs the images in increasing id order (single-process inference)")
        nb = max(1, min(len(self.ids), n_blocks or 2 * max(1, num_procs)))
        self.bounds = np.linspace(0, len(self.ids), nb + 1).round().astype(np.int64)
        self.done, self.next_block, self.buf, self.tasks, self.n_dets = 0, 0, [], [], {}
        self.pool = None
        if num_procs > 1:
            self.pool = _get_pool(gt_path, num_procs)
        else:
            _G["gt"] = self.gt_main
            _G["img_ids"] = self.ids.tolist()

    def add(self, dets, image_ids):
        """dets (n,7) of the next images of the order (image_ids: all of them, also those without detections)."""
        image_ids = [int(i) for i in image_ids]
        if image_ids != self.order[self.done:self.done + len(image_ids)]:
            raise ValueError("images arrived out of order")
        self.done += len(image_ids)
        dets = np.asarray(dets, dtype=np.float64).reshape(-1, 7)
        if dets.shape[0]:
            self.buf.append(dets)
        self._dispatch(final=False)

    def _dispatch(self, final):
        while self.next_block < len(self.bounds) - 1 and (final or self.done >= self.bounds[self.next_block + 1]):
            b = self.next_block
            imgs = self.ids[self.bounds[b]:self.bounds[b + 1]]
            allb = np.concatenate(self.buf) if self.buf else np.zeros((0, 7))
            inb = _in_sorted_ids(allb[:, 0].astype(np.int64), imgs)
            blk, self.buf = allb[inb], ([allb[~inb]] if (~inb).any() else [])
            for c in self.cats:
                sel = blk[blk[:, 6] == c["id"]]
                self.n_dets[c["id"]] = self.n_dets.get(c["id"], 0) + sel.shape[0]
                args = (c["id"], b, imgs, sel)
                self.tasks.append(self.pool.apply_async(_eval_chunk, (args,)) if self.pool else _eval_chunk(args))
            self.next_block += 1

    def finish(self):
        self._dispatch(final=True)
        try:
            chunks = [t.get() if self.pool else t for t in self.tasks]
        except BaseException:
            if self.pool is not None:
                _shutdown_pool(terminate=True)
            raise
        cat_ids = [c["id"] for c in self.cats]
        by_cat = {}
        for c, part, imgs, slim in chunks:
            by_cat.setdefault(c, []).append((part, imgs, slim))
        ids_list = self.ids.tolist()
        results = {}
        for c in cat_ids:
            if self.n_dets.get(c, 0) == 0:  # as evaluate_aitod: no detection of this category at all
                results[c] = _empty_result(self.gt_main, c, img_ids=ids_list)
            else:
                results[c] = _accumulate_category(c, by_cat[c], ids_list)
        prec = np.concatenate([results[c][0] for c in cat_ids], axis=2)
        rec = np.concatenate([results[c][1] for c in cat_ids], axis=1)
        lrp = np.concatenate([results[c][2] for c in cat_ids], axis=0)
        per_class = {}
        for k, c in enumerate(self.cats):
            s = prec[:, :, k, 0, -1]
            v = s[s > -1]
            per_class[c["name"]] = {"AP": float(v.mean()) if v.size else -1.0, "oLRP": float(lrp[k, 0])}
        return {"stats": summarize(prec, rec, lrp), "per_class": per_class}


def summarize(prec, rec, lrp):
    def ap(iou=None, area="all", md=1500):
        s = prec[:, :, :, AREA_LBL.index(area), MAX_DETS.index(md)]
        if iou is not None:
            s = s[np.isclose(IOU_THRS, iou)]
        v = s[s > -1]
        return float(v.mean()) if v.size else -1.0

    def ar(area="all", md=1500):
        s = rec[:, :, AREA_LBL.index(area), MAX_DETS.index(md)]
        v = s[s > -1]
        return float(v.mean()) if v.size else -1.0

    def lr(col):
        v = lrp[:, col]
        v = v[v > -1]
        return float(v.mean()) if v.size else -1.0

    return {
        "AP": ap(), "AP50": ap(0.5), "AP75": ap(0.75),
        "APvt": ap(area="verytiny"), "APt": ap(area="tiny"), "APs": ap(area="small"), "APm": ap(area="medium"),
        "AR1": ar(md=1), "AR100": ar(md=100), "AR1500": ar(md=1500),
        "ARvt": ar("verytiny"), "ARt": ar("tiny"), "ARs": ar("small"), "ARm": ar("medium"),
        "oLRP": lr(0), "oLRP_loc": lr(1), "oLRP_FP": lr(2), "oLRP_FN": lr(3),
    }


def results_table(stats, extra=None):
    """Markdown table in the Dome-DETR layout (paper numbers are quoted, not reproduced)."""
    rows = [
        ("Dome-DETR-M (paper)", "23.9M", "252.6", 34.0, 68.4, 29.9, 18.4, 34.3, 39.0, 46.9),
        ("Dome-DETR-L (paper)", "36.0M", "358.7", 34.6, 69.2, 32.0, 19.0, 35.6, 39.8, 47.3),
    ]
    e = extra or {}
    ours = (e.get("name", "CARE-Tiny (ours)"), e.get("params", "-"), e.get("gflops", "-"),
            *[100 * stats[k] if stats[k] > -1 else None for k in ("AP", "AP50", "AP75", "APvt", "APt", "APs", "APm")])
    lines = ["| Model | Params | GFLOPs | AP | AP50 | AP75 | APvt | APt | APs | APm |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows + [ours]:
        vals = ["-" if x is None else f"{x:.1f}" for x in r[3:]]
        lines.append("| " + " | ".join([str(r[0]), str(r[1]), str(r[2])] + vals) + " |")
    return "\n".join(lines)
