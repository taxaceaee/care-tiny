"""Training / evaluation engine: DDP, AMP, EMA, the Dome-DETR / D-FINE test protocol, resumable checkpoints, logging.

Throughput notes:
  * shapes: K is drawn once per step for all ranks and the denoising block is bucketed in the model, so cuDNN
    autotunes a handful of shapes once;
  * no per-step host syncs in the loop: non-finite steps are skipped by GradScaler (fused AdamW reads found_inf on
    the device), the non-finite counter and the time budget are reduced across ranks every log_interval steps;
  * data: persistent workers (rebuilt only when the augmentation stage changes), photometric augmentation on the GPU,
    a mid-epoch resume skips the consumed batches in the sampler instead of decoding them.
"""

import copy
import functools
import glob
import json
import math
import os
import random
import shutil
import signal
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import utils
from .aitod_eval import StreamingEvaluator, _shutdown_pool, evaluate_aitod
from .config import to_dict
from .data import AITODv2, EpochSampler, RepeatFactorSampler, ShardSampler, collate, photometric_batch, repeat_factors
from .model import compile_hot_modules, postprocess


# ----------------------------------------------------------------------------- EMA
class ModelEMA:
    """EMA of the weights with a D-FINE / YOLO style warm-up: decay * (1 - exp(-updates / warmup))."""

    def __init__(self, model, decay, warmup):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay, self.warmup, self.updates = decay, warmup, 0
        self._cache = None  # (ema tensors, model tensors, model): both are updated in place, no per-step state_dict()

    def _pairs(self, model):
        if self._cache is None or self._cache[2] is not model:
            msd = model.state_dict()
            ema, src = [], []
            for k, v in self.module.state_dict().items():
                if v.dtype.is_floating_point:
                    ema.append(v)
                    src.append(msd[k].detach())
            self._cache = (ema, src, model)
        return self._cache[0], self._cache[1]

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = self.decay * (1 - math.exp(-self.updates / self.warmup))
        ema, src = self._pairs(model)
        # same arithmetic as v.mul_(d).add_(src, alpha=1 - d) per tensor, in a few multi-tensor kernels
        torch._foreach_mul_(ema, d)
        torch._foreach_add_(ema, src, alpha=1 - d)

    def state_dict(self):
        return {"module": self.module.state_dict(), "updates": self.updates}

    def load_state_dict(self, sd):
        self.module.load_state_dict(sd["module"])
        self.updates = sd["updates"]
        self._cache = None


# ----------------------------------------------------------------------------- Dome-DETR / D-FINE protocol
class DFineSelector:
    """Checkpoint selection and stage-2 control of D-FINE's DetSolver.fit, which Dome-DETR trains with
    (github.com/RicePasteM/Dome-DETR, src/solver/det_solver.py). There the "val" loader is the AI-TOD-v2 test split:
    the EMA is scored on test after every epoch and the paper reports the best of these scores. The branches below
    are the solver's, quirks included (unit-tested against a transcription of the solver and the Dome-DETR-M log):

      * a new global best saves best_stg1 (aug stage) or best_stg2 (no-aug stage);
      * at the first no-aug epoch the full state of best_stg1 is reloaded (Trainer);
      * in the no-aug stage, an epoch that is neither a new global best nor the best since the last reset counts
        towards the patience; when it runs out, the local best is reset and the best stage-1 state reloaded;
      * the epoch right after a reset (and any epoch that beats the post-reset best) skips the patience check.
    update() returns (save, reload): save in {None, "stg1", "stg2"}, reload True = reload the best stage-1 state.
    """

    def __init__(self, patience=6):
        self.patience = patience
        self.top1, self.top1_epoch = 0.0, -1  # global best
        self.best_ap, self.best_epoch = None, -1  # best since the last reset
        self.not_improved = 0
        self.history = []  # (epoch, ap)

    def update(self, epoch, ap, stage2):
        self.history.append((epoch, ap))
        save, reload = None, False
        if self.best_ap is not None:
            if ap > self.best_ap:
                self.best_epoch = epoch
            self.best_ap = max(self.best_ap, ap)
        else:
            self.best_ap, self.best_epoch = ap, epoch
        if self.best_ap > self.top1:
            self.top1, self.top1_epoch, self.not_improved = self.best_ap, epoch, 0
            save = "stg2" if stage2 else "stg1"
        else:
            self.not_improved += 1
        if self.best_epoch == epoch:
            if not stage2:
                self.top1 = max(ap, self.top1)
                save = "stg1"
        elif stage2 and self.not_improved >= self.patience:
            self.best_ap, self.best_epoch, self.not_improved = None, -1, 0
            reload = True
        return save, reload

    def reset_stage2(self):
        """train.stage2_reset_patience: the no-aug stage starts with a fresh local best and patience counter."""
        self.best_ap, self.best_epoch, self.not_improved = None, -1, 0

    def state_dict(self):
        return {k: getattr(self, k) for k in ("patience", "top1", "top1_epoch", "best_ap", "best_epoch",
                                               "not_improved", "history")}

    def load_state_dict(self, sd):
        for k, v in sd.items():
            setattr(self, k, v)


# ----------------------------------------------------------------------------- optimiser / schedule
def build_optimizer(model, tc):
    """AdamW with four groups: backbone / rest x with / without weight decay (norms, biases, priors, positions)."""
    groups = {"bb": [], "bb_nowd": [], "other": [], "other_nowd": []}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        nowd = p.ndim <= 1 or any(s in n for s in ("pos", "prior", "norm", "bias"))
        key = ("bb" if n.startswith("backbone.") else "other") + ("_nowd" if nowd else "")
        groups[key].append(p)
    lr, mult, wd = tc["lr"], tc["backbone_lr_mult"], tc["weight_decay"]
    params = [
        {"params": groups["bb"], "lr": lr * mult, "weight_decay": wd, "base_lr": lr * mult},
        {"params": groups["bb_nowd"], "lr": lr * mult, "weight_decay": 0.0, "base_lr": lr * mult},
        {"params": groups["other"], "lr": lr, "weight_decay": wd, "base_lr": lr},
        {"params": groups["other_nowd"], "lr": lr, "weight_decay": 0.0, "base_lr": lr},
    ]
    params = [g for g in params if g["params"]]
    # fused AdamW: one kernel per step, and GradScaler.step() skips inf / NaN steps on the device
    fused = all(p.is_cuda for g in params for p in g["params"])
    return torch.optim.AdamW(params, lr=lr, betas=tuple(tc["betas"]), weight_decay=wd, fused=True if fused else None)


def restore_optimizer_mode(opt):
    """Optimizer.load_state_dict takes 'fused' from the checkpoint; a fused optimizer needs its 'step' tensors on the
    parameter device."""
    fused = bool(getattr(opt, "_step_supports_amp_scaling", False))
    for g in opt.param_groups:
        g["fused"] = True if fused else None
        if fused:
            g["foreach"] = None
        for p in g["params"]:
            st = opt.state.get(p, {})
            if torch.is_tensor(st.get("step")):
                if fused or g.get("capturable"):
                    st["step"] = st["step"].to(device=p.device, dtype=torch.float32)
                else:
                    st["step"] = st["step"].to(device="cpu", dtype=torch.float32)


def lr_factor(step, total, tc):
    """Linear warm-up, then flat for flat_frac of the iterations, then cosine down to min_lr_ratio."""
    w = tc["warmup_iters"]
    if w > 0 and step < w:
        return 0.001 + (1 - 0.001) * step / w
    flat = int(tc["flat_frac"] * total)
    if step < flat:
        return 1.0
    prog = (step - flat) / max(1, total - flat)
    return tc["min_lr_ratio"] + (1 - tc["min_lr_ratio"]) * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))


def stage_lr_factor(step, total, tc, strong):
    """lr_factor, scaled by stage2_lr_mult in the no-aug stage."""
    f = lr_factor(step, total, tc)
    return f if strong else f * float(tc["stage2_lr_mult"])


def stage_ema_decay(tc, strong):
    """EMA decay of a stage: stage2_ema_decay (> 0) in the no-aug stage, else ema_decay."""
    s2 = float(tc["stage2_ema_decay"] or 0.0)
    return s2 if (not strong and s2 > 0) else tc["ema_decay"]


def set_lr(opt, f):
    for g in opt.param_groups:
        g["lr"] = g["base_lr"] * f


def k_from_gt(targets, mcfg, kmax):
    """Training K: the inference rule bucket(k_scale * n + k_offset) with n the GT count of the densest image of the
    batch, so the training K follows the K the model gets at test time. Identical on every rank (max over ranks)."""
    n = max([int(t["boxes"].shape[0]) for t in targets] + [0])
    if utils.is_dist():
        nt = torch.tensor(float(n), device=targets[0]["boxes"].device)
        torch.distributed.all_reduce(nt, op=torch.distributed.ReduceOp.MAX)
        n = int(nt.item())
    need = mcfg["k_scale"] * n + mcfg["k_offset"]
    for b in mcfg["k_buckets"]:
        if b >= need:
            return min(int(b), kmax)
    return min(int(mcfg["k_buckets"][-1]), kmax)


# ----------------------------------------------------------------------------- data
def build_train_loader(ds, sampler, batch_per_gpu, workers, prefetch, pad_div, seed, pad_to=0):
    g = torch.Generator()
    g.manual_seed(seed)  # worker base seed (augmentation streams)
    return DataLoader(ds, batch_size=batch_per_gpu, sampler=sampler, drop_last=True, num_workers=workers,
                      pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0,
                      prefetch_factor=prefetch if workers > 0 else None, worker_init_fn=utils.worker_init_fn,
                      generator=g, collate_fn=functools.partial(collate, pad_divisor=pad_div, pad_to=pad_to))


def build_eval_loader(ds, batch, workers, pad_div, prefetch=2, persistent=False):
    sampler = ShardSampler(len(ds), utils.rank(), utils.world())
    return DataLoader(ds, batch_size=batch, sampler=sampler, num_workers=workers,
                      pin_memory=torch.cuda.is_available(), persistent_workers=bool(persistent and workers > 0),
                      prefetch_factor=prefetch if workers > 0 else None,
                      collate_fn=functools.partial(collate, pad_divisor=pad_div))


def shutdown_loader(loader):
    it = getattr(loader, "_iterator", None) if loader is not None else None
    if it is not None and hasattr(it, "_shutdown_workers"):
        it._shutdown_workers()


def move_targets(targets, device):
    return [{**t, "boxes": t["boxes"].to(device, non_blocking=True),
             "labels": t["labels"].to(device, non_blocking=True)} for t in targets]


# ----------------------------------------------------------------------------- evaluation
@torch.no_grad()
def infer_dets(model, ds, cfg, device, amp=True, sink=None):
    """Inference on this rank's shard of ds -> (dets (n,7) [image_id, x, y, w, h, score, category_id], number of
    images, seconds). sink: a StreamingEvaluator fed batch by batch instead (nothing is kept here)."""
    ec = cfg["eval"]
    model.eval()
    batch = max(1, ec["batch_size"] // utils.world())
    workers = cfg["data"]["num_workers"]
    pad_div = cfg["data"]["pad_divisor"]
    prefetch = cfg["data"].get("prefetch_factor", 2)
    lkey = (batch, workers, pad_div, utils.rank(), utils.world())
    if getattr(ds, "_persistent_eval", False) and workers > 0:
        if getattr(ds, "_eval_loader", None) is None or getattr(ds, "_eval_loader_key", None) != lkey:
            shutdown_loader(getattr(ds, "_eval_loader", None))
            ds._eval_loader = build_eval_loader(ds, batch, workers, pad_div, prefetch, persistent=True)
            ds._eval_loader_key = lkey
        loader = ds._eval_loader
    else:
        loader = build_eval_loader(ds, batch, workers, pad_div, prefetch, persistent=False)
    dets, t0, n_img = [], time.time(), 0
    use_amp = amp and device.type == "cuda"
    # the training autocast dtype (a bf16-trained model is evaluated in bf16)
    amp_dtype = torch.bfloat16 if cfg["train"]["amp_dtype"] == "bf16" else torch.float16
    cat_ids = torch.tensor(ds.cat_ids, dtype=torch.float32, device=device)
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            out = model(images)
        res = postprocess(out, [t["orig_size"] for t in targets], ec)
        rows = []
        for t, r in zip(targets, res):
            b = r["boxes"].float()
            if b.shape[0] == 0:
                continue
            rows.append(torch.cat([torch.full_like(b[:, :1], float(t["image_id"])), b[:, :2], b[:, 2:] - b[:, :2],
                                   r["scores"].float()[:, None], cat_ids[r["labels"]][:, None]], 1))
        batch_dets = torch.cat(rows).cpu().numpy().astype(np.float64) if rows else np.zeros((0, 7), dtype=np.float64)
        if sink is not None:
            sink.add(batch_dets, [t["image_id"] for t in targets])
        elif rows:  # one device -> host copy per batch
            dets.append(batch_dets)
        n_img += len(targets)
    del loader
    return (np.concatenate(dets) if dets else np.zeros((0, 7))), n_img, time.time() - t0


@torch.no_grad()
def evaluate(model, ds, cfg, device, out_dir, tag, logger, amp=True):
    """Distributed inference (every image once) + AI-TOD evaluation on rank 0, written to eval_<tag>.json.
    One process with eval.stream_scoring: the matching of each image block runs in the scoring pool while inference
    goes on (StreamingEvaluator, same result as evaluate_aitod)."""
    ec = cfg["eval"]
    if utils.world() == 1 and ec["stream_scoring"]:
        sink = StreamingEvaluator(ds.ann_path, ds.image_ids, ec["eval_procs"])
        try:
            _, n_img, infer_s = infer_dets(model, ds, cfg, device, amp, sink=sink)
        except BaseException:
            if sink.pool is not None:
                _shutdown_pool(terminate=True)
            raise
        t1 = time.time()
        result = sink.finish()
        result.update({"split": ds.split, "num_images": n_img, "num_dets": int(sum(sink.n_dets.values())),
                       "infer_seconds": infer_s, "eval_seconds": time.time() - t1, "tag": tag, "streamed": True})
        _report_eval(result, out_dir, tag, logger)
        return result
    local, n_img, infer_s = infer_dets(model, ds, cfg, device, amp)
    if utils.world() == 1:
        all_dets, n_all = local, n_img
    else:
        np.save(os.path.join(out_dir, f"_dets_{tag}_rank{utils.rank()}.npy"), local)  # file gather: no huge pickles
        utils.barrier()
        n_all = sum(utils.all_gather_object(n_img))
        all_dets = None
        if utils.is_main():
            paths = sorted(glob.glob(os.path.join(out_dir, f"_dets_{tag}_rank*.npy")))
            parts = [np.load(p) for p in paths]
            all_dets = np.concatenate(parts) if parts else np.zeros((0, 7))
            del parts
            for p in paths:
                os.remove(p)
    result = None
    if utils.is_main():
        t1 = time.time()
        result = evaluate_aitod(ds.ann_path, all_dets, ec["eval_procs"], img_ids=ds.image_ids)
        result.update({"split": ds.split, "num_images": n_all, "num_dets": int(all_dets.shape[0]),
                       "infer_seconds": infer_s, "eval_seconds": time.time() - t1, "tag": tag})
        _report_eval(result, out_dir, tag, logger)
    utils.barrier()
    return result


def _report_eval(result, out_dir, tag, logger):
    with open(os.path.join(out_dir, f"eval_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    s = result["stats"]
    logger.info(f"[eval {tag}] " + " ".join(f"{k}={100 * v:.2f}" for k, v in s.items() if v > -1)
                + f" (infer {result['infer_seconds']:.0f}s, eval {result['eval_seconds']:.0f}s"
                + (", scored during inference)" if result.get("streamed") else ")"))
    logger.log({"type": "eval", "tag": tag, **s, "infer_seconds": result["infer_seconds"],
                "eval_seconds": result["eval_seconds"]})


# ----------------------------------------------------------------------------- training
class Trainer:
    """Owns the datasets, optimiser, EMA and the training loop. Checkpoints (in <out>/checkpoints):
        last.pth           full state of the last epoch (resume)
        ema_last.pth       EMA weights of the last epoch
        best_stg1.pth      full state of the best stage-1 epoch on test (protocol; reloaded at the stage switch)
        best_stg2.pth      full state of the best no-aug epoch on test (train.stage2_reload_best_stg2)
        ema_best_test.pth  EMA weights of the best test epoch (protocol)
        ema_best_val.pth   EMA weights of the best val epoch (train.val_interval)
        ema_epochNNN.pth   EMA snapshots (train.save_every_epochs)
    """

    def __init__(self, cfg, model, criterion, device, out_dir, logger):
        self.cfg, self.model, self.criterion, self.device = cfg, model, criterion, device
        self.out_dir, self.logger = out_dir, logger
        tc = cfg["train"]
        self.tc = tc
        self.ckpt_dir = os.path.join(out_dir, "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        dc = cfg["data"]
        self.dc = dc
        root = dc["root"]
        self.gpu_photo = bool(dc["gpu_photometric"])
        self.train_ds = AITODv2(root, dc["train_split"], train=True, aug=dict(dc["aug"]),
                                max_images=dc["max_train_images"], seed=tc["seed"], gpu_photometric=self.gpu_photo)
        self.val_ds = AITODv2(root, dc["val_split"], max_images=dc["max_val_images"]) \
            if dc["val_split"] and tc["val_interval"] > 0 else None
        # Dome-DETR protocol: the EMA is scored on the test split after every epoch from test_eval_from
        self.test_during_train = bool(tc["test_during_train"])
        assert self.test_during_train or not tc["stage2_reload"], "stage2_reload needs test_during_train"
        self.test_ds = AITODv2(root, dc["test_split"], max_images=dc["max_test_images"]) \
            if self.test_during_train else None
        for ds in (self.val_ds, self.test_ds):
            if ds is not None:
                ds._persistent_eval = True
                ds._eval_loader = None
                ds._eval_loader_key = None
        self.selector = DFineSelector(int(tc["stage2_patience"])) if self.test_during_train else None
        self.stage2_started = False  # the stage-switch reload is done
        self.reloads = []  # (epoch, reason, epoch of the reloaded state)
        self.batch_per_gpu = tc["batch_size"] // utils.world()
        assert self.batch_per_gpu * utils.world() == tc["batch_size"], "batch_size must divide by the world size"
        self.rfs = repeat_factors(self.train_ds, dc["rfs_thr"]) if dc["rfs_thr"] > 0 else None
        if criterion.class_weight_pow > 0:
            src = "legacy constant"
            if criterion.counts_from_data:
                labs = [lab for lab in self.train_ds.labels if len(lab)]
                cnt = np.bincount(np.concatenate(labs), minlength=criterion.C)[:criterion.C] if labs else \
                    np.ones(criterion.C)
                criterion.set_class_counts(cnt)
                src = f"{dc['train_split']} box counts {cnt.tolist()}"
            cw = criterion.class_weights(torch.device("cpu"))
            logger.info(f"class weights ({src}): {[round(float(w), 3) for w in cw]}")
        epoch_len = int(round(self.rfs.sum())) if self.rfs is not None else len(self.train_ds)
        self.iters_per_epoch = epoch_len // (self.batch_per_gpu * utils.world())
        self.total_iters = self.iters_per_epoch * tc["epochs"]
        raw = model.module if hasattr(model, "module") else model
        raw.dn_per_image = tc["dn_per_image"]
        raw.dn_bucket = tc["dn_bucket"]
        self.opt = build_optimizer(raw, tc)
        self.amp = bool(tc["amp"]) and device.type == "cuda"
        self.amp_dtype = torch.bfloat16 if tc["amp_dtype"] == "bf16" else torch.float16
        if self.amp and self.amp_dtype == torch.bfloat16:
            # bf16 needs no loss scaling, but a GradScaler fixed at scale 1 (growth never happens, a non-finite step
            # only halves it, both exact) lets fused AdamW skip non-finite steps on the device without a host sync
            self.scaler = torch.amp.GradScaler("cuda", init_scale=1.0, growth_interval=2 ** 30)
        else:
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp and self.amp_dtype == torch.float16)
        self.ema = ModelEMA(raw, tc["ema_decay"], tc["ema_warmup"])
        if tc["compile"] and device.type == "cuda":
            # after the EMA copy: evaluation stays eager
            compile_hot_modules(raw, tc["compile_mode"], backbone=tc["compile_backbone"])
        self.start_epoch, self.start_iter, self.step, self.bad_steps = 0, 0, 0, 0
        self.best_val_ap = 0.0
        self.t_start = time.time()
        self._loader, self._loader_strong, self._sampler = None, None, None
        self.resume_ckpt_dir = None
        self._stop_signal = False

    # ------------------------------------------------------------------ checkpoints
    def _state(self, epoch_done, iter_done=0, finished=False):
        """Full training state (resume / best_stg1 / best_stg2)."""
        raw = self.model.module if hasattr(self.model, "module") else self.model
        rng = {"py": random.getstate(), "np": np.random.get_state(), "torch": torch.get_rng_state()}
        if self.device.type == "cuda":
            rng["cuda"] = torch.cuda.get_rng_state(self.device)
        return {"model": raw.state_dict(), "ema": self.ema.state_dict(), "opt": self.opt.state_dict(),
                "scaler": self.scaler.state_dict(), "epoch": epoch_done, "iter": iter_done, "step": self.step,
                "bad_steps": self.bad_steps, "best_val_ap": self.best_val_ap, "cfg": to_dict(self.cfg),
                "finished": finished, "dfine": self.selector.state_dict() if self.selector is not None else None,
                "stage2_started": self.stage2_started, "reloads": list(self.reloads), "rng": rng}

    def _save_ema(self, name, epoch, **extra):
        """EMA weights + plain config: loadable with torch.load(weights_only=True) (care_tiny.checkpoint)."""
        utils.atomic_save({"model": self.ema.module.state_dict(), "cfg": to_dict(self.cfg), "epoch": epoch, **extra},
                          os.path.join(self.ckpt_dir, name))

    def save(self, epoch_done, iter_done=0, finished=False):
        if not utils.is_main():
            return
        utils.atomic_save(self._state(epoch_done, iter_done, finished), os.path.join(self.ckpt_dir, "last.pth"))
        self._save_ema("ema_last.pth", epoch_done)

    def try_resume(self, resume):
        """resume: "auto" (<out>/checkpoints/last.pth if present), a checkpoint path, or "" (none)."""
        path = None
        if resume == "auto":
            local = os.path.join(self.ckpt_dir, "last.pth")
            path = local if os.path.isfile(local) else None
        elif resume:
            path = resume
        if not path:
            self.logger.info("no checkpoint to resume, training from scratch (pretrained backbone)")
            return
        self.resume_ckpt_dir = os.path.dirname(os.path.abspath(path))
        if self.resume_ckpt_dir != os.path.abspath(self.ckpt_dir):
            if utils.is_main():  # the protocol checkpoints of the run continue in this output directory
                for pat in ("best_stg1.pth", "best_stg2.pth", "ema_best_test.pth", "ema_best_val.pth"):
                    for src in glob.glob(os.path.join(self.resume_ckpt_dir, pat)):
                        dst = os.path.join(self.ckpt_dir, os.path.basename(src))
                        if not os.path.exists(dst):
                            shutil.copy2(src, dst)
            utils.barrier()
        sd = torch.load(path, map_location="cpu", weights_only=False)
        raw = self.model.module if hasattr(self.model, "module") else self.model
        raw.load_state_dict(sd["model"])
        self.ema.load_state_dict(sd["ema"])
        self.opt.load_state_dict(sd["opt"])
        restore_optimizer_mode(self.opt)
        if sd.get("scaler"):  # empty when saved by a disabled (fp32 / CPU) GradScaler: keep the fresh one
            self.scaler.load_state_dict(sd["scaler"])
        self.start_epoch = int(sd["epoch"])
        self.start_iter = int(sd.get("iter", 0))
        self.step = int(sd["step"])
        self.bad_steps = int(sd.get("bad_steps", 0))
        self.best_val_ap = float(sd.get("best_val_ap", 0.0))
        if self.selector is not None and sd.get("dfine"):
            self.selector.load_state_dict(sd["dfine"])
            self.selector.patience = int(self.tc["stage2_patience"])
        self.stage2_started = bool(sd.get("stage2_started", False))
        self.reloads = list(sd.get("reloads", []))
        if "rng" in sd:
            random.setstate(sd["rng"]["py"])
            np.random.set_state(sd["rng"]["np"])
            torch.set_rng_state(sd["rng"]["torch"].cpu().to(torch.uint8))
            if self.device.type == "cuda" and "cuda" in sd["rng"]:
                torch.cuda.set_rng_state(sd["rng"]["cuda"].cpu().to(torch.uint8), self.device)
        iter_msg = f" (iter {self.start_iter}/{self.iters_per_epoch})" if self.start_iter > 0 else ""
        self.logger.info(f"resumed from {path}: next epoch {self.start_epoch}{iter_msg}, step {self.step}")

    # ------------------------------------------------------------------ Dome-DETR / D-FINE protocol
    def _reload_best(self, epoch, why, prefer_stg2=False):
        """D-FINE's load_resume_state(best_stg1.pth): model, EMA (weights and update count), optimizer, GradScaler and,
        unless stage2_no_lr_rewind, the step counter. With prefer_stg2 and stage2_reload_best_stg2, best_stg2.pth is
        reloaded when it exists."""
        ckpt_name = "best_stg1.pth"
        if prefer_stg2 and self.tc["stage2_reload_best_stg2"]:
            if os.path.isfile(os.path.join(self.ckpt_dir, "best_stg2.pth")) or (
                    self.resume_ckpt_dir and os.path.isfile(os.path.join(self.resume_ckpt_dir, "best_stg2.pth"))):
                ckpt_name = "best_stg2.pth"
        path = os.path.join(self.ckpt_dir, ckpt_name)
        if not os.path.isfile(path) and self.resume_ckpt_dir:
            alt = os.path.join(self.resume_ckpt_dir, ckpt_name)
            if os.path.isfile(alt):
                path = alt
        utils.barrier()  # rank 0 wrote it
        if not os.path.isfile(path):
            self.logger.info(f"[dfine] {why} at epoch {epoch + 1}: no {ckpt_name}, continuing from the current state")
            return
        sd = torch.load(path, map_location="cpu", weights_only=False)
        raw = self.model.module if hasattr(self.model, "module") else self.model
        raw.load_state_dict(sd["model"])
        self.ema.load_state_dict(sd["ema"])
        self.opt.load_state_dict(sd["opt"])
        restore_optimizer_mode(self.opt)
        if sd.get("scaler"):
            self.scaler.load_state_dict(sd["scaler"])
        if not self.tc["stage2_no_lr_rewind"]:
            self.step = int(sd["step"])
        self.reloads.append((epoch + 1, why, int(sd["epoch"])))
        self.logger.info(f"[dfine] {why} at epoch {epoch + 1}: reloaded {ckpt_name} (end of epoch {sd['epoch']}, "
                         f"step {self.step})")

    def _test_protocol(self, epoch, strong):
        """Score the EMA on the test split (tools/test.py numerics), then apply the D-FINE selection: best_stg1.pth /
        best_stg2.pth (full state) and ema_best_test.pth on a new best, and the patience reload in the no-aug stage."""
        tc = self.tc
        with utils.eval_numerics():
            res = evaluate(self.ema.module, self.test_ds, self.cfg, self.device, self.out_dir,
                           f"test_ep{epoch + 1:03d}", self.logger, tc["amp"])
        ap = utils.all_gather_object(float(res["stats"]["AP"]) if res is not None else None)[0]
        save, reload = self.selector.update(epoch + 1, ap, stage2=not strong)
        if save is not None and utils.is_main():
            self._save_ema("ema_best_test.pth", epoch + 1, test_ap=ap)
            if save == "stg1":
                utils.atomic_save(self._state(epoch + 1), os.path.join(self.ckpt_dir, "best_stg1.pth"))
            elif save == "stg2" and tc["stage2_reload_best_stg2"]:
                utils.atomic_save(self._state(epoch + 1), os.path.join(self.ckpt_dir, "best_stg2.pth"))
        s = self.selector
        if save:
            note = f"new best ({save})"
        else:
            note = "patience ran out" if reload else f"patience {s.not_improved}/{s.patience}"
        self.logger.info(f"[dfine] epoch {epoch + 1} test AP {100 * ap:.2f} | best {100 * s.top1:.2f} at epoch "
                         f"{s.top1_epoch} | {note}")
        # no reload after the last epoch: nothing is trained afterwards, and ema_last must stay this epoch's EMA
        if reload and tc["stage2_reload"] and epoch + 1 < tc["epochs"]:
            self._reload_best(epoch, f"patience {s.patience} without a new best", prefer_stg2=True)
        if utils.is_main():
            hist = s.history
            with open(os.path.join(self.out_dir, "dfine_protocol.json"), "w", encoding="utf-8") as f:
                json.dump({"protocol": "D-FINE / Dome-DETR: EMA scored on test after every epoch, best reported",
                           "best_ap": s.top1, "best_epoch": s.top1_epoch, "last_ap": hist[-1][1],
                           "last_epoch": hist[-1][0], "reloads": self.reloads, "history": hist}, f, indent=2)

    def _val(self, epoch):
        res = evaluate(self.ema.module, self.val_ds, self.cfg, self.device, self.out_dir, f"val_ep{epoch + 1:03d}",
                       self.logger, self.tc["amp"])
        if utils.is_main() and res is not None:
            ap = float(res["stats"]["AP"])
            if ap > self.best_val_ap:
                self.best_val_ap = ap
                self.logger.info(f"new best val AP {100 * ap:.2f} at epoch {epoch + 1}")
                self._save_ema("ema_best_val.pth", epoch + 1, val_ap=ap)

    # ------------------------------------------------------------------ data
    def _train_loader(self, epoch, strong, skip_batches):
        """Persistent loader reused across epochs; rebuilt only when the augmentation stage changes, because the
        workers hold their own copy of the dataset and would not see ``strong`` change."""
        if self._loader is None or self._loader_strong != strong:
            self.close()
            self.train_ds.strong = strong
            if self.rfs is not None:
                self._sampler = RepeatFactorSampler(self.rfs, utils.world(), utils.rank(), self.tc["seed"])
            else:
                self._sampler = EpochSampler(self.train_ds, utils.world(), utils.rank(), self.tc["seed"])
            self._loader = build_train_loader(self.train_ds, self._sampler, self.batch_per_gpu, self.dc["num_workers"],
                                              self.dc["prefetch_factor"], self.dc["pad_divisor"],
                                              self.tc["seed"] + 7919 * utils.rank() + 104729 * epoch,
                                              int(self.dc["pad_to"]))
            self._loader_strong = strong
        self._sampler.set_epoch(epoch)
        self._sampler.skip = skip_batches * self.batch_per_gpu
        return self._loader

    def close(self):
        """Stop the persistent train and eval workers."""
        shutdown_loader(self._loader)
        self._loader, self._loader_strong = None, None
        for ds in (self.val_ds, self.test_ds):
            if ds is not None and getattr(ds, "_eval_loader", None) is not None:
                shutdown_loader(ds._eval_loader)
                ds._eval_loader = None
                ds._eval_loader_key = None

    # ------------------------------------------------------------------ loop
    def _sync_status(self, margin_s, bad_local):
        """One all-reduce for both cross-rank decisions: time budget / signal left and total non-finite losses."""
        b = self.tc["time_budget_hours"]
        in_time = b <= 0 or (time.time() - self.t_start) < b * 3600 - margin_s
        ok = 1.0 if in_time and not self._stop_signal else 0.0
        flags = torch.stack([torch.tensor(ok, device=self.device), bad_local.float()])
        flags = utils.all_reduce_mean(flags) * utils.world()
        vals = flags.tolist()
        return vals[0] > utils.world() - 1e-3, int(round(vals[1]))

    def time_left_ok(self, margin_s):
        return self._sync_status(margin_s, torch.zeros((), device=self.device))[0]

    def train(self):
        """Run (or continue) training. Returns True when all epochs are done, False when stopped resumably (time
        budget or SIGTERM)."""
        prev_sigterm = None
        try:
            if hasattr(signal, "SIGTERM"):
                prev_sigterm = signal.getsignal(signal.SIGTERM)
                signal.signal(signal.SIGTERM, lambda signum, frame: setattr(self, "_stop_signal", True))
            return self._train()
        finally:
            if prev_sigterm is not None:
                signal.signal(signal.SIGTERM, prev_sigterm)
            self.close()

    def _train(self):
        tc = self.tc
        epochs = tc["epochs"]
        stage_aug_end = epochs - tc["no_aug_epochs"]
        log_int = tc["log_interval"]
        epoch_time = None
        raw = self.model.module if hasattr(self.model, "module") else self.model
        bad_local = torch.zeros((), device=self.device)  # non-finite losses on this rank (read every log_int steps)
        bad_base = self.bad_steps
        for epoch in range(self.start_epoch, epochs):
            strong = epoch < stage_aug_end
            start_it = self.start_iter if epoch == self.start_epoch else 0
            if epoch == stage_aug_end and not self.stage2_started:
                if self.selector is not None and tc["stage2_reset_patience"]:
                    self.selector.reset_stage2()
                if tc["stage2_reload"]:
                    # D-FINE: the no-aug stage starts from the best stage-1 state (by test AP)
                    self._reload_best(epoch, "stage switch")
                self.stage2_started = True
            if start_it > 0:
                self.logger.info(f"skipping first {start_it} iters of epoch {epoch} (resumed mid-epoch, no decoding)")
            # EMA decay of this stage (set every epoch: the decay is not in the checkpoint, so a resume gets it too)
            self.ema.decay = stage_ema_decay(tc, strong)
            loader = self._train_loader(epoch, strong, start_it)
            self.model.train()
            t_ep, t_data, n_seen, n_done, meter_vec, names = time.time(), 0.0, 0, 0, None, []
            t_last = time.time()
            for i, (images, targets) in enumerate(loader):
                it = start_it + i
                t_data += time.time() - t_last
                if tc["max_iters"] and self.step >= tc["max_iters"]:
                    break
                set_lr(self.opt, stage_lr_factor(self.step, self.total_iters, tc, strong))
                images = images.to(self.device, non_blocking=True)
                targets = move_targets(targets, self.device)
                if self.gpu_photo and (strong or "photometric" in self.train_ds.stage2_keep):
                    photometric_batch(images, targets)
                k = k_from_gt(targets, raw.cfg, tc["train_k_max"])
                with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp):
                    out = self.model(images, targets, k_train=k)
                loss, parts = self.criterion(out, targets)
                if not self.scaler.is_enabled():
                    # fp32 (CPU): no GradScaler to drop a NaN step, so check synchronously (all ranks skip together)
                    finite = utils.all_reduce_mean(torch.isfinite(loss).float().detach())
                    if finite.item() < 1.0:
                        self.bad_steps += 1
                        self.logger.info(f"non-finite loss at step {self.step} (skipped, {self.bad_steps} total): "
                                         f"{ {k_: float(v) for k_, v in parts.items()} }")
                        self.opt.zero_grad(set_to_none=True)
                        if self.bad_steps > 50:
                            raise FloatingPointError("too many non-finite losses")
                        t_last = time.time()
                        continue
                else:
                    # every rank always runs backward (DDP stays in lockstep); a NaN loss gives non-finite
                    # all-reduced grads on every rank and GradScaler skips that step everywhere
                    bad_local += (~torch.isfinite(loss.detach())).float()
                self.opt.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                if tc["clip_grad"] > 0:
                    self.scaler.unscale_(self.opt)
                    gnorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), tc["clip_grad"])
                else:
                    gnorm = torch.zeros((), device=self.device)
                self.scaler.step(self.opt)
                self.scaler.update()
                self.ema.update(raw)
                self.step += 1
                n_seen += images.shape[0] * utils.world()
                names = sorted(parts)
                vec = torch.stack([parts[k_].detach().float() for k_ in names] + [loss.detach().float()])
                meter_vec = vec if meter_vec is None else meter_vec + vec  # stays on the device: no per-step sync
                n_done += 1
                if it % log_int == 0:  # same iteration on every rank: one small all-reduce, then host values
                    margin = 1.2 * (epoch_time or 0) / max(1, self.iters_per_epoch) * log_int + 900
                    time_ok, bad_total = self._sync_status(margin, bad_local)
                    self.bad_steps = bad_base + bad_total
                    if bad_total > 50:
                        raise FloatingPointError(f"too many non-finite losses ({bad_total})")
                    if not time_ok:
                        reason = "SIGTERM received" if self._stop_signal else "time budget reached"
                        self.logger.info(f"{reason} inside epoch {epoch} at iter {it + 1}/{self.iters_per_epoch}; "
                                         f"saving resumable checkpoint")
                        self.save(epoch, iter_done=it + 1, finished=False)
                        return False
                if self.step % log_int == 0 and utils.is_main():
                    el = time.time() - t_ep
                    cur = vec.tolist()
                    rec = {"type": "train", "epoch": epoch, "iter": it, "step": self.step,
                           "lr": self.opt.param_groups[-1]["lr"], "k": k, "loss": cur[-1],
                           "grad_norm": float(gnorm), "img_per_s": n_seen / max(el, 1e-6),
                           "data_frac": t_data / max(el, 1e-6), "nonfinite": self.bad_steps,
                           "mem_gb": torch.cuda.max_memory_allocated(self.device) / 2 ** 30
                           if self.device.type == "cuda" else 0.0,
                           **dict(zip(names, cur[:-1]))}
                    self.logger.log(rec)
                    self.logger.info(f"ep {epoch} it {it}/{self.iters_per_epoch} step {self.step} loss {cur[-1]:.4f} "
                                     f"lr {rec['lr']:.2e} {rec['img_per_s']:.1f} img/s data {rec['data_frac']:.0%} "
                                     f"mem {rec['mem_gb']:.1f}G")
                t_last = time.time()
            raw_ep_time = time.time() - t_ep
            epoch_time = raw_ep_time * (self.iters_per_epoch / max(1, n_done))
            means = {}
            if meter_vec is not None:
                mv = utils.all_reduce_mean(meter_vec / max(1, n_done)).tolist()
                means = dict(zip(names + ["loss"], mv))
            _, bad_total = self._sync_status(0, bad_local)
            self.bad_steps = bad_base + bad_total
            row = {"epoch": epoch, "strong_aug": strong, "steps": self.step, "epoch_minutes": raw_ep_time / 60,
                   "img_per_s": n_seen / max(raw_ep_time, 1e-6), "skipped_steps": self.bad_steps, **means}
            self.logger.epoch_row(row)
            self.logger.info(f"epoch {epoch} done in {raw_ep_time / 60:.1f} min, mean loss {row.get('loss', 0):.4f}")
            if tc["save_every_epochs"] and (epoch + 1) % tc["save_every_epochs"] == 0 and \
                    epoch + 1 >= tc["save_from_epoch"] and utils.is_main():
                self._save_ema(f"ema_epoch{epoch + 1:03d}.pth", epoch + 1)
            if tc["max_iters"] and self.step >= tc["max_iters"]:
                self.save(epoch + 1)
                return True
            # evaluations first, then last.pth: it then holds this epoch's selection state (and a reload), and a
            # session lost before the save repeats the epoch instead of resuming past an evaluation that never ran
            if self.val_ds is not None and ((epoch + 1) % tc["val_interval"] == 0 or epoch + 1 == epochs):
                self._val(epoch)
            if self.test_ds is not None and epoch + 1 >= int(tc["test_eval_from"]):
                self._test_protocol(epoch, strong)
            self.save(epoch + 1)
            if not self.time_left_ok(1.3 * epoch_time + 600):
                self.logger.info("time budget / stop signal: not starting another epoch; stopping (resumable)")
                return epoch + 1 >= epochs
        return True
