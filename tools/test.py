"""Evaluate a checkpoint on an AI-TOD-v2 split with the AI-TOD protocol (maxDets 1500, very tiny / tiny / small /
medium AP, oLRP).

    python tools/test.py --checkpoint weights/care_tiny_b2_aitodv2_ep115.pth --data-root /data/aitodv2
    torchrun --nproc_per_node=2 tools/test.py --checkpoint ... --data-root ... --split test

Writes eval_<tag>.json (all metrics, per-class AP / oLRP, timing) and results_table_<tag>.md to --output-dir.
The numerics are those of the in-training test protocol: EMA weights, the training autocast dtype, channels_last on
CUDA, TF32 matmuls and cuDNN autotuning off.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if sys.platform.startswith("linux"):
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from care_tiny import utils  # noqa: E402
from care_tiny.aitod_eval import results_table  # noqa: E402
from care_tiny.checkpoint import load_model  # noqa: E402
from care_tiny.data import AITODv2  # noqa: E402
from care_tiny.engine import evaluate  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-root", required=True, help="AI-TOD-v2 folder (annotations/ and images/)")
    ap.add_argument("--split", default="test", choices=["train", "val", "trainval", "test"])
    ap.add_argument("--output-dir", default="runs/eval")
    ap.add_argument("--tag", default="")
    ap.add_argument("--opts", nargs="*", default=[], help="config overrides, e.g. eval.batch_size=8")
    ap.add_argument("--trusted", action="store_true", help="allow full training states (pickled RNG states)")
    args = ap.parse_args()

    device, _ = utils.init_distributed()
    model, cfg, ckpt = load_model(args.checkpoint, device, args.opts, trusted=args.trusted)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
        model.channels_last = True
    os.makedirs(args.output_dir, exist_ok=True)
    logger = utils.Logger(args.output_dir)
    ds = AITODv2(args.data_root, args.split)
    tag = args.tag or f"{args.split}_{os.path.splitext(os.path.basename(args.checkpoint))[0]}"
    with utils.eval_numerics():
        res = evaluate(model, ds, cfg, device, args.output_dir, tag, logger, cfg["train"]["amp"])
    if utils.is_main() and res is not None:
        n_params = sum(p.numel() for p in model.parameters())
        table = results_table(res["stats"], {"params": f"{n_params / 1e6:.2f}M",
                                             "name": f"CARE-Tiny (epoch {ckpt.get('epoch')})"})
        per_class = "\n".join(f"| {k} | {100 * v['AP']:.1f} | {100 * v['oLRP']:.1f} |"
                              for k, v in res["per_class"].items())
        with open(os.path.join(args.output_dir, f"results_table_{tag}.md"), "w", encoding="utf-8") as f:
            f.write(table + "\n\n| Class | AP | oLRP |\n|---|---:|---:|\n" + per_class + "\n")
        print(table)
        print(json.dumps({k: round(100 * v, 2) for k, v in res["stats"].items()}, indent=1))
    logger.close()
    utils.barrier()
    if utils.is_dist():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
