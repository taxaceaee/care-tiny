"""Configuration: nested defaults, YAML overlays and ``section.key=value`` command-line overrides.

The defaults below are the recipe of the released AI-TOD-v2 model (``configs/care_tiny_b2_aitodv2.yaml`` spells
them out). YAML files are merged in order, then overrides; an unknown key is an error, so a typo never silently
falls back to a default.
"""

import copy
import json

import yaml

DEFAULTS = {
    "model": {
        # same backbone and pretrained weights as D-FINE / Dome-DETR-M (HGNetv2-B2, SSLD stage 1)
        "backbone": "hgnetv2_b2.ssld_stage1_in22k_in1k",
        "pretrained": True,
        "freeze_bn": True,
        "freeze_stem": False,
        "backbone_levels": 4,
        # neck: top-down FPN + bottom-up PAN, ConvNeXt fusion blocks per level (stride 4, 8, 16, 32)
        "neck_blocks": [2, 1, 1, 1],
        "neck_dim": 128,
        "head_convs": 2,  # depth of the decoupled cls / box towers of the dense head
        # shared stride-2 detail map F2 (HiResMap) read by every query window
        "hires_dim": 32,
        "hires_blocks": 1,
        "hires_kernel": 7,
        # stride-1 pixel map channels and PAN semantic channels, concatenated to F2 in every window cell
        "pix_dim": 8,
        "sem_dim": 64,
        "sem_obj_cells": 4.0,  # semantic level: the PAN level where the object spans 4..8 feature pixels
        # query windows: window x window cells, cell size sigma = clip(max(w, h) / window_obj, 1, 2**(levels-1)) px
        "window": 16,
        "window_obj": 12.0,
        "pyramid_levels": 6,
        "query_dim": 192,
        "heads": 6,
        "scene_layers": 1,  # transformer layers over the 64 scene tokens
        "ctx_dim": 0,  # channels of the shared stride-8/16 context projection (0 = query_dim)
        "num_readouts": 2,  # cascade: one-to-many readouts, the last one one-to-one after the KNN attention
        "num_classes": 8,
        "knn": 16,
        # candidate routing: M candidates (spatial quota + global top), K of them refined
        "num_candidates": 4096,
        "quota_regions": 16,
        "quota_per_region": 4,
        "k_buckets": [128, 256, 512, 1024, 2048],
        "k_scale": 2.0,  # K bucket = bucket(k_scale * n + k_offset), n = object count (GT in training)
        "k_scale_eval": 4.0,  # k_scale at inference, n = objectness peak count
        "k_offset": 64,
        "count_thr": 0.3,
        "spatial_frac": 0.25,  # share of K reserved for spatial-quota region winners
        "region_min_prob": 0.05,  # region winners below this objectness get no reserved K slot
        "select_score_alpha": 0.35,  # routing score (1 - a) * objectness logit + a * max class logit
        "recenter_sig_ema": 0.5,  # recentred window: cell size damped towards the previous window, grown by overflow
        "edge_overflow_scale": 4.0,  # learned residual (cells) letting an edge leave the window
        "nwd_const": 12.8,
        "small_thr": 16.0,  # GTs below sqrt(wh) = small_thr are matched with NWD, larger ones with IoU
    },
    "loss": {
        "dense_obj": 1.0,
        "dense_cls": 1.0,
        "dense_l1": 1.0,
        "dense_giou": 2.0,
        "dense_nwd": 1.0,
        "ref_cls": 1.0,
        "ref_l1": 1.0,
        "ref_giou": 2.0,
        "ref_nwd": 1.0,
        "ref_dfl": 0.25,
        "dn_weight": 1.0,
        "tal_alpha": 1.0,
        "tal_beta": 6.0,
        "dense_topk": 9,
        "dense_radius": 8,
        "o2m_topk": 4,
        # one-to-one readout: 2 = GTs that lost their best query to another GT take their best free query
        "o2o_passes": 2,
        "qfl_beta": 2.0,
        # score target of a positive: w * IoU + (1 - w) * size-aware metric (NWD for small GTs)
        "hybrid_iou_weight": 0.7,
        # off-peak positive sub-slots get objectness target (1 - g) + g * q / q_max (sharp peaks)
        "dense_obj_quality": 0.5,
        # class weight ((1 / C) / f_c) ** p, normalised so that sum_c f_c * w_c = 1
        "class_weight_pow": 0.15,
        # box counts f_c of the class weights: "legacy" = the constant the released model was trained with (not the
        # AI-TOD-v2 counts, see criterion.AITOD_BOX_COUNTS_LEGACY); "data" = the train split's own labels (corrected
        # weighting, not trained at the full schedule)
        "class_counts": "legacy",
        "qual_loss_weight": 0.5,  # direct quality supervision of the readout quality head
    },
    "data": {
        "root": "",
        "train_split": "trainval",
        "val_split": "",
        "test_split": "test",
        "pad_divisor": 32,
        # > 0: training batches padded to pad_to x pad_to (one static shape for torch.compile)
        "pad_to": 800,
        "num_workers": 8,
        "prefetch_factor": 4,
        # photometric parameters are drawn in the loader workers, the pixel work runs on the GPU batch
        "gpu_photometric": True,
        "max_train_images": 0,
        "max_val_images": 0,
        "max_test_images": 0,
        # repeat-factor sampling threshold (LVIS); 0 = uniform sampling
        "rfs_thr": 0.1,
        "aug": {
            "photometric_p": 0.5,
            "brightness": 0.2,
            "contrast": 0.2,
            "saturation": 0.2,
            "hue": 0.02,
            "scale_p": 0.5,
            "scale_range": [0.8, 1.25],
            "hflip_p": 0.5,
            "vflip_p": 0.5,
            "rot90_p": 0.5,
            "min_size": 0.5,
            "min_visible": 0.5,
            # copy-paste of rare-class instances (classes in < copy_paste_img_frac of the images)
            "copy_paste_p": 0.5,
            "copy_paste_img_frac": 0.1,
            "copy_paste_max": 8,
            # strong-stage ops kept in the no-aug stage (copy_paste, photometric, scale, rot90); flips always run
            "stage2_keep": ["copy_paste"],
        },
    },
    "train": {
        "epochs": 160,
        "no_aug_epochs": 40,  # the last epochs run without the strong augmentations
        "batch_size": 16,  # global batch over all GPUs
        "lr": 2.0e-4,
        "backbone_lr_mult": 0.5,
        "weight_decay": 1.0e-4,
        "betas": [0.9, 0.999],
        "warmup_iters": 1000,
        # flat-cosine schedule: constant for flat_frac of the iterations, then cosine down to min_lr_ratio
        "flat_frac": 0.5,
        "min_lr_ratio": 0.05,
        "clip_grad": 0.1,
        "amp": True,
        "amp_dtype": "bf16",  # bf16 (Ampere and newer) | fp16 (with loss scaling)
        "ema_decay": 0.9998,
        "ema_warmup": 2000,
        "train_k_max": 1024,  # upper bound of the training K (bucket of the densest image of the batch)
        "dn_per_image": 100,
        # denoising slots are padded to a multiple of this, so the query count takes few distinct shapes
        "dn_bucket": 25,
        "seed": 0,
        "log_interval": 50,
        "val_interval": 0,  # > 0: score the EMA on data.val_split every N epochs and after the last one
        "save_every_epochs": 0,  # > 0: keep ema_epochNNN.pth snapshots
        "save_from_epoch": 0,
        "time_budget_hours": 0.0,  # > 0: stop cleanly (resumable) before this much wall time has passed
        "channels_last": True,
        "compile": True,  # torch.compile of the neck, heads and readouts (CUDA only; evaluation stays eager)
        "compile_mode": "default",
        "compile_backbone": True,
        "final_test": True,  # score the final EMA on data.test_split (skipped if the protocol already did)
        # Dome-DETR / D-FINE protocol (engine.DFineSelector): the EMA is scored on the test split after every epoch
        # from test_eval_from (1-based), the best epoch is reported, and the no-aug stage starts from the best
        # stage-1 state (stage2_reload) and reloads it after stage2_patience epochs without a new best
        "test_during_train": True,
        "test_eval_from": 101,
        "stage2_reload": True,
        "stage2_patience": 6,
        "stage2_reset_patience": True,  # patience counter starts at 0 in the no-aug stage
        "stage2_no_lr_rewind": True,  # a reload keeps the step counter (no jump back up the LR schedule)
        "stage2_reload_best_stg2": True,  # patience reloads use the best no-aug state once there is one
        "stage2_lr_mult": 0.5,  # LR multiplier in the no-aug stage
        "stage2_ema_decay": 0.9994,  # EMA decay in the no-aug stage (0 = ema_decay)
        "max_iters": 0,  # > 0: stop after this many iterations (smoke tests)
    },
    "eval": {
        "max_dets": 1500,
        "score_thr": 0.001,
        "pre_nms_topk": 3000,
        "nms_iou": 0.6,
        # refined class scores fused with the dense class scores of the same sub-slots: rs ** (1 - a) * dense ** a
        "score_fuse": 0.25,
        # unrefined candidates: score * coarse_score_mult * sigmoid(objectness) ** coarse_obj_pow
        "coarse_score_mult": 0.95,
        "coarse_obj_pow": 0.15,
        "batch_size": 16,
        "eval_procs": 8,  # AI-TOD scoring processes (the result does not depend on it)
        # single process: score finished image blocks while inference goes on (same result, less wall time)
        "stream_scoring": True,
    },
}


class Cfg(dict):
    """dict with attribute access for nested configs."""

    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return Cfg(v) if isinstance(v, dict) and not isinstance(v, Cfg) else v

    def __setattr__(self, k, v):
        self[k] = v


def _merge(base, upd, path=""):
    for k, v in upd.items():
        if k not in base:
            raise KeyError(f"unknown config key: {path + k}")
        if isinstance(base[k], dict) and isinstance(v, dict):
            _merge(base[k], v, path + k + ".")
        else:
            base[k] = v
    return base


def _to_cfg(d):
    return Cfg({k: _to_cfg(v) if isinstance(v, dict) else v for k, v in d.items()})


def parse_overrides(overrides):
    """["train.lr=1e-4", ...] -> nested dict (values parsed as YAML)."""
    out = {}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override must be key=value, got {item!r}")
        key, val = item.split("=", 1)
        node = out
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    return out


def load_config(paths=None, overrides=(), base=None):
    """Defaults (or ``base``, e.g. the config stored in a checkpoint), then the YAML file(s) ``paths`` in order,
    then the ``key=value`` overrides."""
    cfg = copy.deepcopy(DEFAULTS)
    if base is not None:
        _merge(cfg, copy.deepcopy(dict(base)))
    for p in [p for p in ([paths] if isinstance(paths, str) else list(paths or ())) if p]:
        with open(p, encoding="utf-8") as f:
            _merge(cfg, yaml.safe_load(f) or {})
    _merge(cfg, parse_overrides(overrides))
    return _to_cfg(cfg)


def to_dict(cfg):
    """Plain nested dict (stored in checkpoints, which then load with torch.load(weights_only=True))."""
    return json.loads(json.dumps(cfg))


def dump_config(cfg, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
