import json
import os
import random
from unittest import mock

import numpy as np
import pytest
import torch

from care_tiny import utils
from care_tiny.aitod_eval import evaluate_aitod
from care_tiny.checkpoint import export_weights, load_model
from care_tiny.criterion import Criterion
from care_tiny.engine import DFineSelector, Trainer, lr_factor, stage_ema_decay, stage_lr_factor
from care_tiny.model import CareTiny, postprocess

from conftest import small_cfg, synthetic_batch

# Test AP after each epoch (0..150) of Dome-DETR-M on AI-TOD-v2, from its official log
# (github.com/RicePasteM/Dome-DETR, logs/Dome-M-AITOD-20260109_111347.log, "current_stat")
DOME_M_TEST_AP = [
    0.048381, 0.09077, 0.126898, 0.14321, 0.149701, 0.155664, 0.16103, 0.165307, 0.169441, 0.173808, 0.179117, 0.18322,
    0.188643, 0.194099, 0.198783, 0.204436, 0.209971, 0.213901, 0.218913, 0.223954, 0.228203, 0.233497, 0.237146,
    0.240355, 0.243302, 0.246362, 0.249341, 0.252429, 0.254993, 0.257049, 0.259346, 0.26276, 0.264473, 0.266683,
    0.268891, 0.27089, 0.272838, 0.27541, 0.276156, 0.277258, 0.278764, 0.279562, 0.281023, 0.282178, 0.282381, 0.28238,
    0.284359, 0.285922, 0.287097, 0.287227, 0.288029, 0.289326, 0.289785, 0.291074, 0.292642, 0.292724, 0.294379,
    0.297154, 0.297834, 0.298088, 0.297751, 0.298377, 0.298244, 0.297305, 0.297202, 0.296677, 0.296237, 0.29705,
    0.297455, 0.298971, 0.298894, 0.300197, 0.300219, 0.301914, 0.30193, 0.301991, 0.301344, 0.301515, 0.301951,
    0.302801, 0.303405, 0.304209, 0.304142, 0.305039, 0.306238, 0.306521, 0.307132, 0.308187, 0.309099, 0.310064,
    0.310345, 0.311102, 0.311128, 0.312339, 0.312357, 0.311915, 0.312909, 0.313209, 0.313164, 0.313417, 0.312661,
    0.313759, 0.312999, 0.313176, 0.313861, 0.314135, 0.315394, 0.316368, 0.317394, 0.31951, 0.321973, 0.324399,
    0.326092, 0.326499, 0.329209, 0.330519, 0.331017, 0.331086, 0.331782, 0.333033, 0.333705, 0.334177, 0.334867,
    0.335473, 0.335598, 0.335591, 0.335766, 0.335777, 0.336781, 0.337143, 0.337674, 0.338019, 0.339047, 0.338359,
    0.33878, 0.339038, 0.33966, 0.339477, 0.339394, 0.339284, 0.339288, 0.339062, 0.338027, 0.31769, 0.320311, 0.322331,
    0.323677, 0.324838, 0.326209, 0.32741, 0.329027]


def dfine_reference(aps, stop_epoch, patience=6):
    """Literal transcription of D-FINE / Dome-DETR DetSolver.fit's selection (one metric): per epoch "stg1" / "stg2"
    (checkpoint saved), "reload" (best_stg1 reloaded), "tolerate:<count>" or None."""
    best_stat, top1, not_improved, out, k = {"epoch": -1}, 0, 0, [], "coco_eval_bbox"
    for epoch, ap in enumerate(aps):
        act = None
        if k in best_stat:
            best_stat["epoch"] = epoch if ap > best_stat[k] else best_stat["epoch"]
            best_stat[k] = max(best_stat[k], ap)
        else:
            best_stat["epoch"] = epoch
            best_stat[k] = ap
        if best_stat[k] > top1:
            not_improved = 0
            top1 = best_stat[k]
            act = "stg2" if epoch >= stop_epoch else "stg1"
        else:
            not_improved += 1
        if best_stat["epoch"] == epoch:
            if epoch >= stop_epoch:
                if ap > top1:
                    top1 = ap
                    act = "stg2"
            else:
                top1 = max(ap, top1)
                act = "stg1"
        elif epoch >= stop_epoch:
            if not_improved >= patience:
                best_stat = {"epoch": -1}
                not_improved = 0
                act = "reload"
            else:
                act = f"tolerate:{not_improved}"
        out.append(act)
    return out


def selector_actions(aps, stop_epoch, patience=6):
    s, out = DFineSelector(patience), []
    for epoch, ap in enumerate(aps):
        save, reload = s.update(epoch, ap, stage2=epoch >= stop_epoch)
        wait = epoch >= stop_epoch and save is None and not reload and s.best_epoch != epoch
        out.append("reload" if reload else save or (f"tolerate:{s.not_improved}" if wait else None))
    return out, s


def test_selector_replays_the_dome_detr_m_log():
    acts, s = selector_actions(DOME_M_TEST_AP, stop_epoch=108)
    assert acts == dfine_reference(DOME_M_TEST_AP, 108)
    # the log shows "Tolerate" at epochs 125, 133-135, 137-141 and a reload of best_stg1 at 142 only
    tol = [(e, int(a.split(":")[1])) for e, a in enumerate(acts) if a and a.startswith("tolerate")]
    assert tol == [(125, 1), (133, 1), (134, 2), (135, 3), (137, 1), (138, 2), (139, 3), (140, 4), (141, 5)]
    assert [e for e, a in enumerate(acts) if a == "reload"] == [142]
    assert (s.top1_epoch, round(s.top1, 6)) == (136, 0.33966)  # the paper's 34.0


def test_selector_matches_the_solver_on_random_runs():
    rng = random.Random(0)
    for _ in range(300):
        n, stop, a, aps = rng.randint(1, 60), rng.randint(0, 40), rng.random() * 0.2, []
        for _ in range(n):  # rising, falling and flat stretches, with exact ties
            a = round(min(1.0, max(0.0, a + rng.choice([-0.02, -0.005, 0.0, 0.0, 0.005, 0.01, 0.03]))), 3)
            aps.append(a)
        p = rng.randint(1, 7)
        assert selector_actions(aps, stop, p)[0] == dfine_reference(aps, stop, p)


def test_learning_rate_and_ema_schedules():
    tc = {"warmup_iters": 10, "flat_frac": 0.5, "min_lr_ratio": 0.05, "stage2_lr_mult": 0.5, "ema_decay": 0.9998,
          "stage2_ema_decay": 0.9994}
    assert lr_factor(0, 100, tc) == pytest.approx(0.001) and lr_factor(10, 100, tc) == 1.0
    assert lr_factor(49, 100, tc) == 1.0 and lr_factor(100, 100, tc) == pytest.approx(0.05)
    assert lr_factor(75, 100, tc) == pytest.approx(0.05 + 0.95 * 0.5)
    assert stage_lr_factor(80, 100, tc, True) == lr_factor(80, 100, tc)
    assert stage_lr_factor(80, 100, tc, False) == pytest.approx(0.5 * lr_factor(80, 100, tc))
    assert stage_ema_decay(tc, True) == 0.9998 and stage_ema_decay(tc, False) == 0.9994
    assert stage_ema_decay({**tc, "stage2_ema_decay": 0.0}, False) == 0.9998


def test_trainer_runs_the_test_protocol_and_resumes(aitod_root, tmp_path):
    """6 epochs (3 aug + 3 no-aug), patience 1, scripted test APs, the release's stage-2 settings: the no-aug stage
    starts from the best stage-1 epoch (2) with a fresh patience, epoch 5 runs out of patience and reloads it (the
    step counter is kept), epoch 6 is the new best; a resumed trainer gets the selection state back."""
    aps = {1: 0.1, 2: 0.3, 3: 0.2, 4: 0.25, 5: 0.24, 6: 0.35}
    out = str(tmp_path / "run")
    cfg = small_cfg([f"data.root={aitod_root}", "train.batch_size=2", "train.epochs=6", "train.no_aug_epochs=3",
                     "train.test_eval_from=1", "train.stage2_patience=1", "train.warmup_iters=0",
                     "train.ema_warmup=2", "train.log_interval=1", "data.rfs_thr=0.0"])
    seen = []

    def fake_eval(model, ds, cfg_, device, out_dir, tag, logger, amp=True):
        assert not torch.backends.cudnn.benchmark  # scored with the reported numerics
        seen.append(tag)
        return {"stats": {"AP": aps[int(tag[-3:])]}}

    torch.manual_seed(0)
    logger = utils.Logger(out)
    with mock.patch("care_tiny.engine.evaluate", side_effect=fake_eval):
        tr = Trainer(cfg, CareTiny(cfg), Criterion(cfg), torch.device("cpu"), out, logger)
        assert tr.train()
    ipe = tr.iters_per_epoch
    assert seen == [f"test_ep{e:03d}" for e in range(1, 7)]
    assert tr.reloads == [(4, "stage switch", 2), (5, "patience 1 without a new best", 2)]
    assert tr.step == 6 * ipe  # train.stage2_no_lr_rewind: reloads keep the step counter
    ck = os.path.join(out, "checkpoints")
    best = torch.load(os.path.join(ck, "ema_best_test.pth"), map_location="cpu", weights_only=True)
    assert (best["epoch"], best["test_ap"]) == (6, 0.35)
    assert torch.load(os.path.join(ck, "best_stg2.pth"), map_location="cpu", weights_only=False)["epoch"] == 6
    with open(os.path.join(out, "dfine_protocol.json")) as f:
        rep = json.load(f)
    assert (rep["best_epoch"], rep["best_ap"], rep["last_epoch"], rep["last_ap"]) == (6, 0.35, 6, 0.35)
    tr2 = Trainer(cfg, CareTiny(cfg), Criterion(cfg), torch.device("cpu"), out, logger)
    tr2.try_resume("auto")
    assert (tr2.start_epoch, tr2.step, tr2.stage2_started) == (6, 6 * ipe, True)
    assert tr2.selector.state_dict() == tr.selector.state_dict() and tr2.reloads == tr.reloads
    for a, b in zip(tr.ema.module.state_dict().values(), tr2.ema.module.state_dict().values()):
        assert torch.equal(a, b)
    logger.close()

    # export the best EMA as an inference file and load it back (weights_only)
    dst = str(tmp_path / "weights.pth")
    export_weights(os.path.join(ck, "ema_best_test.pth"), dst, note="test")
    model, cfg2, ckpt = load_model(dst)
    assert ckpt["epoch"] == 6 and ckpt["test_ap"] == 0.35 and ckpt["note"] == "test"
    assert cfg2["model"] == {**cfg["model"], "pretrained": False}
    for k, v in best["model"].items():
        assert torch.equal(model.state_dict()[k], v)


def test_training_learns_synthetic_objects(tmp_path):
    """End to end on two images: the loss falls and the boxes are found (AP50 on the training images)."""
    torch.manual_seed(0)
    random.seed(0)
    cfg = small_cfg(["model.num_classes=3"])
    model = CareTiny(cfg)
    model.dn_per_image = 8
    crit = Criterion(cfg)
    imgs, targets = synthetic_batch(2, 128, seed=7)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(150):
        model.train()
        loss, _ = crit(model(imgs, targets, k_train=32), targets)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.detach()))
    model.eval()
    with torch.no_grad():
        res = postprocess(model(imgs), [(128, 128)] * 2, cfg["eval"])
    gt = str(tmp_path / "gt.json")
    anns = [{"id": n + 1, "image_id": t["image_id"], "category_id": int(lab), "iscrowd": 0,
             "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]], "area": (b[2] - b[0]) * (b[3] - b[1])}
            for n, (t, b, lab) in enumerate((t, b, lab) for t in targets
                                            for b, lab in zip(t["boxes"].tolist(), t["labels"].tolist()))]
    with open(gt, "w") as f:
        json.dump({"images": [{"id": t["image_id"], "file_name": "x", "width": 128, "height": 128} for t in targets],
                   "annotations": anns, "categories": [{"id": c, "name": str(c)} for c in range(3)]}, f)
    rows = [[t["image_id"], b[0], b[1], b[2] - b[0], b[3] - b[1], s, lab] for t, r in zip(targets, res)
            for b, s, lab in zip(r["boxes"].tolist(), r["scores"].tolist(), r["labels"].tolist())]
    stats = evaluate_aitod(gt, np.array(rows).reshape(-1, 7), num_procs=1)["stats"]
    assert np.mean(losses[-10:]) < 0.5 * np.mean(losses[:10])
    assert stats["AP50"] > 0.3, stats
