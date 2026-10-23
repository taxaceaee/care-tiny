"""Loading released / trained weights and exporting a training checkpoint as a slim inference file."""

import torch

from .config import load_config, to_dict
from .model import CareTiny


def read_checkpoint(path, trusted=False):
    """Released weights and EMA files hold only tensors and plain containers, so they load with
    ``weights_only=True``. Full training states (last.pth, best_stg1.pth) also hold RNG states and need
    ``trusted=True`` (only for files you produced yourself)."""
    return torch.load(path, map_location="cpu", weights_only=not trusted)


def model_weights(ckpt):
    """EMA weights of a training state, else the stored weights."""
    return ckpt["ema"]["module"] if "ema" in ckpt else ckpt["model"]


def load_model(path, device="cpu", overrides=(), trusted=False):
    """Build the detector from a checkpoint's own config (plus ``key=value`` overrides) and load its weights.
    Returns (model in eval mode on ``device``, config, checkpoint dict)."""
    ckpt = read_checkpoint(path, trusted)
    cfg = load_config(None, ["model.pretrained=false", *overrides], base=ckpt["cfg"])
    model = CareTiny(cfg)
    model.load_state_dict(model_weights(ckpt))
    return model.to(device).eval(), cfg, ckpt


def export_weights(src, dst, trusted=False, **meta):
    """Write the EMA weights and config of ``src`` (any checkpoint written by the trainer) to ``dst`` as fp32 tensors
    and plain containers, with the epoch and any ``meta`` fields (e.g. test_ap)."""
    ckpt = read_checkpoint(src, trusted)
    out = {"model": {k: v.detach().cpu().contiguous() for k, v in model_weights(ckpt).items()},
           "cfg": to_dict(ckpt["cfg"]), "epoch": ckpt.get("epoch")}
    for k in ("test_ap", "val_ap"):
        if k in ckpt:
            out[k] = ckpt[k]
    out.update(meta)
    torch.save(out, dst)
    return out
