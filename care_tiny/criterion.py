"""Training losses (computed in fp32).

Dense head: objectness (Gaussian-weighted focal / quality focal), quality focal class loss, L1 + GIoU + NWD boxes.
Readouts: quality focal class loss on class x quality, L1 + GIoU + NWD boxes and a distribution focal loss on the
edge distributions, for the refined and the denoising queries separately.

Assignment runs batched and the losses use fixed-size masked sums instead of gathers of a variable number of
positives, so a loss call (forward and backward) runs without host syncs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import box_ops
from .assigner import dense_assign_slots, refine_assign_batched
from .utils import all_reduce_mean


def qfl_logits(logits, target, beta):
    p = torch.sigmoid(logits)
    return F.binary_cross_entropy_with_logits(logits, target, reduction="none") * (p - target).abs().pow(beta)


def qfl_prob(p, target, beta):
    p = p.clamp(1e-6, 1 - 1e-6)
    bce = -(target * torch.log(p) + (1 - target) * torch.log(1 - p))
    return bce * (p - target).abs().pow(beta)


def box_losses_elem(pred, gt, const):
    """pred / gt xyxy (...,4) -> per box: L1 normalised by the GT size, 1 - GIoU, 1 - NWD, each (...)."""
    gw = (gt[..., 2] - gt[..., 0]).clamp(min=0)
    gh = (gt[..., 3] - gt[..., 1]).clamp(min=0)
    scale = torch.sqrt(gw * gh).clamp(min=1.0)[..., None]
    l1 = ((pred - gt).abs() / scale).sum(-1)
    ps = box_ops.sort_xyxy(pred)
    return l1, 1 - box_ops.giou_aligned(ps, gt), 1 - box_ops.nwd_aligned(ps, gt, const)


def dfl_elem(logp, target):
    """logp (...,4,W+1) log marginal over the boundaries, target (...,4) continuous in [0, W] -> (...)."""
    nb = logp.shape[-1]
    # nan_to_num: off-mask targets come from the predicted boxes; a NaN there must not become a gather index
    t = torch.nan_to_num(target, nan=0.0).clamp(0, nb - 1 - 1e-3)
    lo = t.floor().long()
    wr = t - lo.float()
    lp_lo = logp.gather(-1, lo[..., None]).squeeze(-1)
    lp_hi = logp.gather(-1, (lo + 1)[..., None]).squeeze(-1)
    return -((1 - wr) * lp_lo + wr * lp_hi).mean(-1)


def edges(b):
    """xyxy (...,4) -> (x1, x2, y1, y2): the order of the edge distributions."""
    return torch.stack([b[..., 0], b[..., 2], b[..., 1], b[..., 3]], -1)


def pad_targets(targets, dev):
    """Padded GT tensors: boxes (B,G,4), labels (B,G), valid (B,G) with G >= 1."""
    B = len(targets)
    counts = [int(t["boxes"].shape[0]) for t in targets]
    G = max(counts + [1])
    gtb = torch.zeros(B, G, 4, device=dev)
    gtl = torch.zeros(B, G, dtype=torch.long, device=dev)
    gval = torch.zeros(B, G, dtype=torch.bool, device=dev)
    for b, (t, n) in enumerate(zip(targets, counts)):
        if n:
            gtb[b, :n] = t["boxes"]
            gtl[b, :n] = t["labels"]
            gval[b, :n] = True
    return gtb, gtl, gval


# Box counts per class (airplane, bridge, storage-tank, ship, swimming-pool, vehicle, person, wind-mill) of the
# official aitodv2_trainval.json: vehicle is 87.5% of the boxes, wind-mill 0.07%.
AITODV2_TRAINVAL_BOX_COUNTS = (843.0, 755.0, 8219.0, 17913.0, 899.0, 329441.0, 18306.0, 249.0)
# The constant the released model was trained with (loss.class_counts=legacy). It is not the AI-TOD-v2 trainval
# count (airplane 14 325 vs 843, storage-tank 23 551 vs 8 219, ship 34 606 vs 17 913), so the released model's
# airplane weight is 1.37 instead of 2.14. Kept to reproduce the released checkpoint exactly.
AITOD_BOX_COUNTS_LEGACY = (14325.0, 1904.0, 23551.0, 34606.0, 842.0, 309555.0, 15716.0, 249.0)


class Criterion(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.lc = dict(cfg["loss"])
        self.mc = dict(cfg["model"])
        self.acfg = {**self.lc, "nwd_const": self.mc["nwd_const"], "small_thr": self.mc["small_thr"]}
        self.W = self.mc["window"]
        self.C = self.mc["num_classes"]
        # score target of a positive: w * IoU + (1 - w) * size-aware metric. The metric keeps a non-zero target for a
        # shifted tiny box (a 4 px box shifted by 4 px has IoU 0 but NWD 0.73), the IoU aligns the ranking with AP
        self.hybrid_w = float(self.lc["hybrid_iou_weight"])
        self.dense_obj_q = float(self.lc["dense_obj_quality"])
        self.class_weight_pow = float(self.lc["class_weight_pow"])
        self.qual_loss_w = float(self.lc["qual_loss_weight"])
        src = self.lc["class_counts"]
        if src not in ("data", "legacy"):
            raise ValueError(f"loss.class_counts must be data or legacy, got {src!r}")
        # data: the Trainer replaces these with the counts of its train split (set_class_counts)
        self.counts_from_data = src == "data"
        base = AITOD_BOX_COUNTS_LEGACY if src == "legacy" else AITODV2_TRAINVAL_BOX_COUNTS
        init_counts = torch.tensor(base[:self.C], dtype=torch.float32) if self.C == 8 else torch.ones(self.C)
        self.register_buffer("cls_counts", init_counts, persistent=False)

    def set_class_counts(self, counts):
        """Box counts per class of the train split (loss.class_counts=data). Returns False (and keeps the constant)
        with class_counts=legacy."""
        if not self.counts_from_data:
            return False
        c = torch.as_tensor(counts, dtype=torch.float32).flatten()
        if c.numel() != self.C:
            raise ValueError(f"set_class_counts: {c.numel()} counts for {self.C} classes")
        self.cls_counts.copy_(c.to(self.cls_counts.device))
        return True

    def _quality_target(self, pred, gt, metric_q=None):
        if metric_q is None:
            metric_q = box_ops.size_aware_metric_aligned(pred, gt, self.mc["nwd_const"], self.mc["small_thr"])
        return self.hybrid_w * box_ops.iou_aligned(pred, gt) + (1.0 - self.hybrid_w) * metric_q

    @torch.no_grad()
    def class_weights(self, dev):
        if self.class_weight_pow <= 0:
            return None
        cnt = self.cls_counts.to(dev).clamp(min=1.0)
        freq = cnt / cnt.sum()
        cw = ((1.0 / self.C) / freq).pow(self.class_weight_pow).clamp(0.80, 2.5)
        return cw / (freq * cw).sum()

    def forward(self, out, targets):
        """Returns (weighted total loss, dict of the unweighted parts)."""
        lc = self.lc
        dev = out["dense"]["obj"].device
        gtb, gtl, gval = pad_targets(targets, dev)
        cw = self.class_weights(dev)
        losses = self._dense(out, gtb, gtl, gval, cw, dev)
        n_ro = len(out["refine"])
        for t, r in enumerate(out["refine"]):
            topk = lc["o2m_topk"] if t < n_ro - 1 else 1  # one-to-many readouts, the last one one-to-one
            for k, v in self._refine(out, r, gtb, gtl, gval, topk, cw, dev).items():
                losses[f"r{t + 1}_{k}"] = v
        w = {"dense_obj": lc["dense_obj"], "dense_cls": lc["dense_cls"], "dense_l1": lc["dense_l1"],
             "dense_giou": lc["dense_giou"], "dense_nwd": lc["dense_nwd"]}
        for t in range(1, n_ro + 1):
            for kind, scale in (("", 1.0), ("dn_", lc["dn_weight"])):
                w[f"r{t}_{kind}cls"] = lc["ref_cls"] * scale
                w[f"r{t}_{kind}l1"] = lc["ref_l1"] * scale
                w[f"r{t}_{kind}giou"] = lc["ref_giou"] * scale
                w[f"r{t}_{kind}nwd"] = lc["ref_nwd"] * scale
                w[f"r{t}_{kind}dfl"] = lc["ref_dfl"] * scale
        total = sum(w[k] * v for k, v in losses.items() if k in w)
        return total, losses

    # ------------------------------------------------------------------ dense
    def _dense(self, out, gtb, gtl, gval, cw, dev):
        d = out["dense"]
        Hs, Ws = d["hw"]
        obj, cls, boxes = d["obj"], d["cls"], box_ops.cxcywh_to_xyxy(d["boxes"])
        B, Ns, C = cls.shape
        cls_prob = torch.sigmoid(cls.detach())
        # every positive sub-slot is exactly one (b, g, j) candidate entry with pos True, so masked sums over the
        # entries equal sums over the positives (no nonzero() / boolean indexing: no host sync)
        keys, pos, gauss = dense_assign_slots(gtb, gtl, gval, cls_prob, boxes.detach(), Hs, Ws, self.acfg)
        k1 = keys.shape[-1]
        flat = keys.reshape(-1)
        posf = pos.float()
        pred = boxes.reshape(B * Ns, 4).index_select(0, flat).view(*keys.shape, 4)
        gt = torch.where(pos[..., None], gtb[:, :, None, :].expand_as(pred), pred.detach())  # finite off the mask
        q = self._quality_target(pred.detach(), gt)
        pos_mask = torch.zeros(B * Ns, device=dev).scatter_reduce(0, flat, posf.reshape(-1), "amax").view(B, Ns) > 0
        lab = gtl[:, :, None].expand(-1, -1, k1).reshape(-1)
        tgt_cls = torch.zeros(B * Ns * C, device=dev).scatter_reduce(0, flat * C + lab, (q * posf).reshape(-1), "amax")
        tgt_cls = tgt_cls.view(B, Ns, C)
        npos = all_reduce_mean(pos_mask.sum().float()).clamp(min=1.0)  # stays on the device
        p = torch.sigmoid(obj).clamp(1e-6, 1 - 1e-6)
        # peak-sharpened objectness: each GT's best-aligned positive keeps target 1, off-peak positives get
        # (1 - g) + g * q / q_max, which turns the 3x3 plateau around an object into a peak
        q_max = (q * posf).amax(-1, keepdim=True).clamp(min=1e-6)
        q_rel = ((1.0 - self.dense_obj_q) + self.dense_obj_q * (q / q_max).clamp(0.0, 1.0)) * posf
        tgt_obj = torch.zeros(B * Ns, device=dev).scatter_reduce(0, flat, q_rel.reshape(-1), "amax").view(B, Ns)
        pos_l = qfl_prob(p, tgt_obj, 2.0)
        neg_l = -((1 - gauss) ** 4) * (p ** 2) * torch.log(1 - p)
        l_obj = torch.where(pos_mask, pos_l, neg_l).sum() / npos
        qfl_c = qfl_logits(cls, tgt_cls, self.lc["qfl_beta"])
        if cw is not None:
            # positive class / box losses weighted by the class weight; background QFL keeps weight 1
            w_cls = torch.where(tgt_cls > 0, cw.view(1, 1, C), torch.ones_like(tgt_cls))
            l_cls = (qfl_c * w_cls).sum() / npos
            w_box = cw[gtl.clamp(0, C - 1)][:, :, None] * posf
        else:
            l_cls = qfl_c.sum() / npos
            w_box = posf
        l1, gi, nw = box_losses_elem(pred, gt, self.mc["nwd_const"])
        return {"dense_obj": l_obj, "dense_cls": l_cls, "dense_l1": (l1 * w_box).sum() / npos,
                "dense_giou": (gi * w_box).sum() / npos, "dense_nwd": (nw * w_box).sum() / npos}

    # ------------------------------------------------------------------ readouts
    def _refine(self, out, r, gtb_pad, gtl_pad, gval, topk, cw, dev):
        q = out["q"]
        Kb = q["Kb"]
        geom = r["geom"]  # the window this readout read
        box, cls, qual, logp = r["box"], r["cls"], r["qual"], r["edge_logp"]
        B, Q, C = cls.shape
        q_sig = torch.sigmoid(qual)[..., None]
        cls_sig = torch.sigmoid(cls)
        prob_assign = (cls_sig.detach() * q_sig.detach())[:, :Kb]
        pboxes = box_ops.sort_xyxy(box.detach())
        o, qq = refine_assign_batched(pboxes[:, :Kb], prob_assign, q["valid"][:, :Kb],
                                      {k: geom[k][:, :Kb] for k in ("X0", "Y0", "sig")}, self.W,
                                      gtb_pad, gtl_pad, gval, topk, self.acfg)
        owner, quality = o, qq
        if q["dn_gt"] is not None and Q > Kb:
            dg = q["dn_gt"]  # (B, nd) index of the noised GT, -1 for padded slots
            dv = q["valid"][:, Kb:]
            dgc = dg.clamp(min=0)
            gt_dn = gtb_pad.gather(1, dgc[..., None].expand(-1, -1, 4))
            qd = box_ops.size_aware_metric_aligned(pboxes[:, Kb:], gt_dn, self.mc["nwd_const"], self.mc["small_thr"])
            owner = torch.cat([o, torch.where(dv, dg, torch.full_like(dg, -1))], 1)
            quality = torch.cat([qq, torch.where(dv, qd, torch.zeros_like(qd))], 1)
        elif Q > Kb:
            owner = torch.cat([o, torch.full((B, Q - Kb), -1, dtype=torch.long, device=dev)], 1)
            quality = torch.cat([qq, torch.zeros(B, Q - Kb, device=dev)], 1)
        own_c = owner.clamp(min=0)
        gt_o = gtb_pad.gather(1, own_c[..., None].expand(-1, -1, 4))
        q_tgt = self._quality_target(pboxes, gt_o, quality)
        quality = torch.where(owner >= 0, q_tgt, torch.zeros_like(quality))
        # per-query losses on all (B, Q) queries, summed under the positive mask of each part
        posq = (owner >= 0) & q["valid"]
        qlab = gtl_pad.gather(1, own_c)
        if self.qual_loss_w > 0:
            # the C - 1 negative classes of a positive query do not pull its quality towards 0 (detached there), and
            # the quality logit is supervised directly with the box quality target
            is_tgt = torch.zeros(B, Q, C, dtype=torch.bool, device=dev).scatter_(2, qlab[..., None], posq[..., None])
            prob = cls_sig * torch.where(is_tgt, q_sig, q_sig.detach())
        else:
            prob = cls_sig * q_sig
        tgt = torch.zeros_like(prob).scatter_(2, qlab[..., None],
                                              torch.where(posq, quality, torch.zeros_like(quality))[..., None])
        qfl = qfl_prob(prob, tgt, self.lc["qfl_beta"]).sum(-1)
        if self.qual_loss_w > 0:
            tgt_q = torch.where(posq, quality, torch.zeros_like(quality))
            qfl = qfl + self.qual_loss_w * qfl_logits(qual, tgt_q, self.lc["qfl_beta"])
        gt_s = torch.where(posq[..., None], gt_o, pboxes)  # finite losses and grads off the mask
        l1e, gie, nwe = box_losses_elem(box, gt_s, self.mc["nwd_const"])
        org = torch.stack([geom["X0"], geom["X0"], geom["Y0"], geom["Y0"]], -1)
        dfe = dfl_elem(logp, (edges(gt_s) - org) / geom["sig"][..., None])
        if cw is not None:
            wq = torch.where(posq, cw[qlab.clamp(0, C - 1)], torch.ones_like(quality))
            qfl = qfl * wq
            l1e, gie, nwe, dfe = l1e * wq, gie * wq, nwe * wq, dfe * wq
        res = {}
        is_dn = torch.zeros(B, Q, dtype=torch.bool, device=dev)
        is_dn[:, Kb:] = True
        for kind, part in (("", ~is_dn), ("dn_", is_dn)):
            m_valid = (part & q["valid"]).float()
            pos = (posq & part).float()
            npos = all_reduce_mean(pos.sum()).clamp(min=1.0)
            res.update({f"{kind}cls": (qfl * m_valid).sum() / npos, f"{kind}l1": (l1e * pos).sum() / npos,
                        f"{kind}giou": (gie * pos).sum() / npos, f"{kind}nwd": (nwe * pos).sum() / npos,
                        f"{kind}dfl": (dfe * pos).sum() / npos})
        return res
