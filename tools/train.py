"""Train CARE-Tiny on AI-TOD-v2 (one GPU or torchrun DDP).

    python tools/train.py --config configs/care_tiny_b2_aitodv2.yaml --data-root /data/aitodv2 --output-dir runs/b2
    torchrun --nproc_per_node=2 tools/train.py --config configs/care_tiny_b2_aitodv2.yaml --data-root ... \
        --output-dir runs/b2 --opts train.batch_size=16

The output directory holds train.log, train_log.jsonl, epoch_summary.csv, eval_*.json, dfine_protocol.json,
results_table.md, status.json and checkpoints/. Re-running the same command resumes from checkpoints/last.pth.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# K and the denoising count change the step shapes; expandable segments keep the caching allocator from
# fragmenting. Must be set before the first CUDA allocation (Linux only).
if sys.platform.startswith("linux"):
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: E402

from care_tiny import utils  # noqa: E402
from care_tiny.aitod_eval import results_table  # noqa: E402
from care_tiny.config import dump_config, load_config  # noqa: E402
from care_tiny.criterion import Criterion  # noqa: E402
from care_tiny.data import AITODv2  # noqa: E402
from care_tiny.engine import Trainer, evaluate  # noqa: E402
from care_tiny.model import CareTiny  # noqa: E402


def best_epoch(selector):
    """Best scored epoch (the solver starts its best at 0: a run scoring 0 everywhere has no "new best")."""
    return selector.top1_epoch if selector.top1_epoch > 0 else max(selector.history, key=lambda h: h[1])[0]


def write_results_table(out, name, selector, n_params):
    """results_table.md: best test epoch (Dome-DETR protocol) and last epoch (official baseline protocol)."""
    tables = []
    for label, ep in (("best test epoch, Dome-DETR protocol", best_epoch(selector)),
                      ("last epoch, official baseline protocol", selector.history[-1][0])):
        path = os.path.join(out, f"eval_test_ep{ep:03d}.json")
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                stats = json.load(f)["stats"]
            tables.append(results_table(stats, {"params": f"{n_params / 1e6:.2f}M",
                                                "name": f"CARE-Tiny {name} (epoch {ep}, {label})"}))
    if tables:
        with open(os.path.join(out, "results_table.md"), "w", encoding="utf-8") as f:
            f.write("\n\n".join(tables) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", nargs="+", default=[], help="YAML files merged in order (default: the built-in recipe)")
    ap.add_argument("--data-root", default="", help="AI-TOD-v2 folder (annotations/ and images/); overrides data.root")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--resume", default="auto", help="'auto' (output-dir/checkpoints/last.pth), a path, or ''")
    ap.add_argument("--opts", nargs="*", default=[], help="config overrides, e.g. train.epochs=12 data.num_workers=4")
    args = ap.parse_args()

    cfg = load_config(args.config, args.opts)
    if args.data_root:
        cfg["data"]["root"] = args.data_root
    device, local = utils.init_distributed()
    utils.seed_everything(cfg["train"]["seed"])
    torch.backends.cudnn.benchmark = True
    # TF32 for the fp32 parts (losses, LayerNorm inputs, grid_sample, fp32 einsums) on Ampere and newer
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    out = args.output_dir
    os.makedirs(out, exist_ok=True)
    logger = utils.Logger(out)
    if utils.is_main():
        dump_config(cfg, os.path.join(out, "config_resolved.json"))
        with open(os.path.join(out, "env.json"), "w", encoding="utf-8") as f:
            json.dump(utils.env_info(), f, indent=2)

    model = CareTiny(cfg).to(device)
    if cfg["train"]["channels_last"] and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
        model.channels_last = True
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"model params {n_params / 1e6:.2f}M (trainable {n_train / 1e6:.2f}M), world {utils.world()}, "
                f"device {device}, data root {cfg['data']['root']}")
    criterion = Criterion(cfg)
    net = DDP(model, device_ids=[local] if device.type == "cuda" else None, find_unused_parameters=False,
              gradient_as_bucket_view=True) if utils.is_dist() else model
    trainer = Trainer(cfg, net, criterion, device, out, logger)
    trainer.try_resume(args.resume)
    logger.info(f"train images {len(trainer.train_ds)}, iters/epoch {trainer.iters_per_epoch}, "
                f"batch/gpu {trainer.batch_per_gpu}, epochs {cfg['train']['epochs']}")
    finished = trainer.train()

    status = {"finished": bool(finished), "step": trainer.step, "time": utils.now(),
              "hours_this_session": (time.time() - trainer.t_start) / 3600, "best_val_ap": trainer.best_val_ap}
    sel = trainer.selector
    last_scored = bool(sel is not None and sel.history and sel.history[-1][0] == cfg["train"]["epochs"])
    if sel is not None and sel.history:
        status["dfine"] = {"best_ap": sel.top1, "best_epoch": best_epoch(sel), "last_ap": sel.history[-1][1],
                           "last_epoch": sel.history[-1][0], "reloads": trainer.reloads}
        if utils.is_main():
            write_results_table(out, os.path.basename(os.path.normpath(out)), sel, n_params)
    # the last epoch's EMA was already scored on test by the protocol: no second pass
    if finished and cfg["train"]["final_test"] and not last_scored:
        test_ds = AITODv2(cfg["data"]["root"], cfg["data"]["test_split"], max_images=cfg["data"]["max_test_images"])
        with utils.eval_numerics():
            res = evaluate(trainer.ema.module, test_ds, cfg, device, out, "test_final", logger, cfg["train"]["amp"])
        if utils.is_main() and res is not None:
            with open(os.path.join(out, "results_table.md"), "w", encoding="utf-8") as f:
                f.write(results_table(res["stats"], {"params": f"{n_params / 1e6:.2f}M",
                                                     "name": "CARE-Tiny (EMA of the last epoch)"}) + "\n")
            status["test"] = res["stats"]
    if utils.is_main():
        with open(os.path.join(out, "status.json"), "w", encoding="utf-8") as f:
            json.dump(status, f, indent=2)
        logger.info(f"status: {status}")
    logger.close()
    utils.barrier()
    if utils.is_dist():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
