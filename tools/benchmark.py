"""Parameters, GFLOPs (torch FlopCounterMode, 2 FLOPs per MAC) and batch-1 latency (P50 / P95, including the
post-processing) at fixed refinement budgets K.

    python tools/benchmark.py --output runs/benchmark.json                      # the built-in recipe
    python tools/benchmark.py --checkpoint weights/care_tiny_b2_aitodv2_ep115.pth --k 256 1024

K is adaptive at inference (from the estimated object count): AI-TOD-v2 test images mostly use K = 128 or 256.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.flop_counter import FlopCounterMode  # noqa: E402

from care_tiny.checkpoint import load_model  # noqa: E402
from care_tiny.config import load_config  # noqa: E402
from care_tiny.model import CareTiny, postprocess  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", nargs="*", default=[])
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--size", type=int, default=800)
    ap.add_argument("--k", type=int, nargs="*", default=[128, 256, 512, 1024])
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--output", default="")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--opts", nargs="*", default=[])
    args = ap.parse_args()
    device = torch.device(args.device)
    if args.checkpoint:
        model, cfg, _ = load_model(args.checkpoint, device, args.opts)
    else:
        cfg = load_config(args.config, ["model.pretrained=false", *args.opts])
        model = CareTiny(cfg).to(device).eval()
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
        model.channels_last = True
    amp_dtype = torch.bfloat16 if cfg["train"]["amp_dtype"] == "bf16" else torch.float16
    res = {"device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu", "size": args.size,
           "params_M": sum(p.numel() for p in model.parameters()) / 1e6,
           "params_backbone_M": sum(p.numel() for p in model.backbone.parameters()) / 1e6, "per_k": {}}
    img = torch.randint(0, 255, (1, 3, args.size, args.size), dtype=torch.uint8, device=device)
    for k in args.k:
        entry = {}
        with FlopCounterMode(display=False) as fc:
            model(img, k_override=k)
        entry["GFLOPs"] = fc.get_total_flops() / 1e9
        if device.type == "cuda":
            times = []
            for i in range(args.warmup + args.iters):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.autocast("cuda", dtype=amp_dtype):
                    out = model(img, k_override=k)
                    postprocess(out, [(args.size, args.size)], cfg["eval"])
                torch.cuda.synchronize()
                if i >= args.warmup:
                    times.append((time.perf_counter() - t0) * 1000)
            t = np.array(times)
            entry.update({"latency_ms_p50": float(np.percentile(t, 50)), "latency_ms_p95": float(np.percentile(t, 95)),
                          "fps": float(1000 / t.mean()), "peak_mem_GB": torch.cuda.max_memory_allocated() / 2 ** 30})
        res["per_k"][str(k)] = entry
        print(f"K={k}: {entry}", flush=True)
    print(json.dumps(res, indent=2))
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
