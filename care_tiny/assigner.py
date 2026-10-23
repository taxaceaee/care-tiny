"""Label assignment, vectorised over the batch (no Hungarian matching, no host syncs).

dense_assign_slots    task-aligned assignment on the stride-2 sub-slot grid; every GT keeps its centre sub-slot
                      as a positive (unique for disjoint GTs: two disjoint boxes with sides >= 2 px never share it).
refine_assign_batched assignment of the refined queries with the same metric s^alpha * m^beta, restricted to GTs whose
                      centre lies inside the query window; topk = 4 for the one-to-many readouts, 1 for the last.

GT tensors are padded to (B, G) with a validity mask; sub-slots / queries of image b are flattened to b * N + i.
"""

import torch

from . import box_ops

BIG = 1.0e4


def _resolve(keys, vals, owners, n_slots, n_owners, ok=None):
    """For each slot keep the pair with the highest value; ties go to the smallest owner id. Returns the owner of
    every slot (-1 = none). ``ok`` masks pairs out with torch.where (boolean indexing would sync with the host)."""
    if ok is not None:
        vals = torch.where(ok, vals, torch.full_like(vals, -float("inf")))
    best = torch.full((n_slots,), -float("inf"), device=vals.device).scatter_reduce(0, keys, vals, "amax")
    win = vals >= best[keys]
    if ok is not None:
        win &= ok
    cand = torch.where(win, owners, torch.full_like(owners, n_owners))
    choice = torch.full((n_slots,), n_owners, device=vals.device, dtype=torch.long)
    choice = choice.scatter_reduce(0, keys, cand, "amin")
    return torch.where(choice < n_owners, choice, torch.full_like(choice, -1))


@torch.no_grad()
def dense_assign_slots(gt, labels, gvalid, cls_prob, pred, Hs, Ws, cfg):
    """gt (B,G,4) xyxy padded, labels (B,G), gvalid (B,G) bool, cls_prob (B,Ns,C), pred (B,Ns,4) xyxy.

    Returns keys (B,G,k+1): flat sub-slot ids b*Ns+i of every GT's candidates (top-k by the TAL metric + centre),
    pos (B,G,k+1) bool: the candidate is a positive of this GT, and gauss (B,Ns): the CenterNet-style Gaussian that
    down-weights negatives around every GT. Every positive sub-slot appears exactly once, so masked sums over
    (B,G,k+1) are sums over the positives.
    """
    keys, _, owner, gauss = _dense_candidates(gt, labels, gvalid, cls_prob, pred, Hs, Ws, cfg)
    G = gt.shape[1]
    own = torch.arange(G, device=gt.device).view(1, G, 1)
    # owner == g implies g has a valid candidate entry for that sub-slot (owners only come from valid entries)
    pos = owner[keys] == own
    # a GT's centre sub-slot can also be one of its top-k: keep only the first occurrence in the row
    k1 = keys.shape[-1]
    earlier = torch.ones(k1, k1, dtype=torch.bool, device=gt.device).tril(-1)  # [j, i]: i < j
    dup = ((keys[..., :, None] == keys[..., None, :]) & earlier).any(-1)
    return keys, pos & ~dup, gauss.view(keys.shape[0], -1)


@torch.no_grad()
def _dense_candidates(gt, labels, gvalid, cls_prob, pred, Hs, Ws, cfg):
    """Candidates of every GT and the owner of every sub-slot (-1 = negative). keys (B,G,k+1), ok (B,G,k+1) =
    valid GT and inside the radius, owner (B*Ns,), gauss (B*Ns,)."""
    dev = pred.device
    B, Ns, C = cls_prob.shape
    G = gt.shape[1]
    gauss = torch.zeros(B * Ns, device=dev)
    if G == 0:
        e = torch.zeros(B, 0, 1, dtype=torch.long, device=dev)
        return e, e.bool(), torch.full((B * Ns,), -1, dtype=torch.long, device=dev), gauss
    Rm = int(cfg["dense_radius"])
    g = box_ops.xyxy_to_cxcywh(gt)
    r = torch.ceil(g[..., 2:].amax(-1) / 4).clamp(2, Rm).long()
    off = torch.arange(-Rm, Rm + 1, device=dev)
    oy, ox = torch.meshgrid(off, off, indexing="ij")
    oy, ox = oy.reshape(-1), ox.reshape(-1)
    cxs = torch.floor(g[..., 0] / 2).long().clamp(0, Ws - 1)
    cys = torch.floor(g[..., 1] / 2).long().clamp(0, Hs - 1)
    xs, ys = cxs[..., None] + ox, cys[..., None] + oy  # (B,G,S)
    inr = (ox.abs() <= r[..., None]) & (oy.abs() <= r[..., None])
    inr &= (xs >= 0) & (xs < Ws) & (ys >= 0) & (ys < Hs) & gvalid[..., None]
    idx = ys.clamp(0, Hs - 1) * Ws + xs.clamp(0, Ws - 1)
    fidx = idx + (torch.arange(B, device=dev) * Ns).view(B, 1, 1)  # flat sub-slot ids
    S = idx.shape[-1]
    s = cls_prob.reshape(B * Ns, C)[fidx, labels[..., None].expand_as(fidx)]
    m = box_ops.size_aware_metric_aligned(pred.reshape(B * Ns, 4)[fidx], gt[:, :, None, :].expand(-1, -1, S, -1),
                                          cfg["nwd_const"], cfg["small_thr"])
    metric = torch.where(inr, s.clamp(min=1e-9) ** cfg["tal_alpha"] * m.clamp(min=1e-9) ** cfg["tal_beta"],
                         torch.full_like(m, -1.0))
    k = min(int(cfg["dense_topk"]), S)
    tv, ti = metric.topk(k, dim=-1)
    centre = Rm * (2 * Rm + 1) + Rm
    cols = torch.cat([ti, torch.full((B, G, 1), centre, device=dev, dtype=torch.long)], -1)
    vals = torch.cat([tv, torch.full((B, G, 1), BIG, device=dev)], -1)
    ok = torch.cat([tv > -1, inr[..., centre:centre + 1]], -1)
    keys = fidx.gather(-1, cols)
    own = torch.arange(G, device=dev).view(1, G, 1).expand_as(cols)
    owner = _resolve(keys.reshape(-1), vals.reshape(-1), own.reshape(-1), B * Ns, G, ok.reshape(-1))
    # Gaussian penalty reduction (CenterNet) for the negatives around every GT
    scx = 2 * xs.float() + 1
    scy = 2 * ys.float() + 1
    sg = torch.clamp(torch.sqrt(g[..., 2] * g[..., 3]) / 6, min=1.0)[..., None]
    gv = torch.exp(-((scx - g[..., 0:1]) ** 2 + (scy - g[..., 1:2]) ** 2) / (2 * sg ** 2))
    gauss = gauss.scatter_reduce(0, fidx.reshape(-1), torch.where(inr, gv, torch.zeros_like(gv)).reshape(-1), "amax")
    return keys, ok, owner, gauss


@torch.no_grad()
def refine_assign_batched(qbox, qprob, qvalid, geom, W, gt, labels, gvalid, topk, cfg):
    """qbox (B,Q,4) xyxy, qprob (B,Q,C), qvalid (B,Q), geom dict of (B,Q) X0 / Y0 / sig, gt (B,G,4) padded,
    labels (B,G), gvalid (B,G). Returns owner (B,Q) (-1 = background) and quality (B,Q)."""
    dev = qbox.device
    B, Q = qbox.shape[:2]
    G = gt.shape[1]
    if G == 0 or Q == 0:
        return torch.full((B, Q), -1, dtype=torch.long, device=dev), torch.zeros(B, Q, device=dev)
    m = box_ops.size_aware_metric_pairwise(qbox, gt, cfg["nwd_const"], cfg["small_thr"])  # (B,Q,G)
    s = qprob.gather(2, labels[:, None, :].expand(-1, Q, -1))
    metric = s.clamp(min=1e-9) ** cfg["tal_alpha"] * m.clamp(min=1e-9) ** cfg["tal_beta"]
    gc = box_ops.xyxy_to_cxcywh(gt)
    X0, Y0, sig = geom["X0"][..., None], geom["Y0"][..., None], geom["sig"][..., None]
    gx, gy = gc[:, None, :, 0], gc[:, None, :, 1]
    inside = (gx >= X0) & (gx < X0 + W * sig) & (gy >= Y0) & (gy < Y0 + W * sig)
    inside &= qvalid[..., None] & gvalid[:, None, :]
    metric = torch.where(inside, metric, torch.full_like(metric, -1.0))
    k = min(int(topk), Q)
    tv, ti = metric.topk(k, dim=1)  # (B,k,G): best queries per GT
    keys = ti + (torch.arange(B, device=dev) * Q).view(B, 1, 1)
    own = torch.arange(G, device=dev).view(1, 1, G).expand_as(ti)
    owner = _resolve(keys.reshape(-1), tv.reshape(-1), own.reshape(-1), B * Q, G, (tv > -1).reshape(-1)).view(B, Q)
    if k == 1:
        # one-to-one: a GT whose best query went to another GT has no positive; each further pass gives those GTs
        # their best query among the free ones
        for _ in range(int(cfg.get("o2o_passes", 1)) - 1):
            taken = owner >= 0
            got = torch.zeros(B, G, device=dev).scatter_reduce(1, owner.clamp(min=0), taken.float(), "amax") > 0
            m2 = torch.where(taken[..., None] | got[:, None, :], torch.full_like(metric, -1.0), metric)
            tv2, ti2 = m2.topk(1, dim=1)
            keys2 = ti2 + (torch.arange(B, device=dev) * Q).view(B, 1, 1)
            o2 = _resolve(keys2.reshape(-1), tv2.reshape(-1), own.reshape(-1), B * Q, G,
                          (tv2 > -1).reshape(-1)).view(B, Q)
            owner = torch.where(taken, owner, o2)
    pos = owner >= 0
    qual = torch.where(pos, m.gather(2, owner.clamp(min=0)[..., None]).squeeze(-1), torch.zeros_like(qbox[..., 0]))
    return owner, qual
