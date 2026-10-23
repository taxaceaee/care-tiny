"""Distributed helpers, seeding, logging, evaluation numerics and atomic checkpoint IO."""

import contextlib
import csv
import datetime
import json
import os
import platform
import random
import subprocess
import sys

import numpy as np
import torch
import torch.distributed as dist


def is_dist():
    return dist.is_available() and dist.is_initialized()


def rank():
    return dist.get_rank() if is_dist() else 0


def world():
    return dist.get_world_size() if is_dist() else 1


def is_main():
    return rank() == 0


def init_distributed():
    """torchrun sets RANK / WORLD_SIZE / LOCAL_RANK. Returns (device, local_rank)."""
    if "RANK" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        local = int(os.environ["LOCAL_RANK"])
        timeout = datetime.timedelta(minutes=90)  # rank 0 scores the evaluation while the others wait at a barrier
        if torch.cuda.is_available():
            torch.cuda.set_device(local)
            dist.init_process_group("nccl", timeout=timeout, device_id=torch.device("cuda", local))
            return torch.device("cuda", local), local
        dist.init_process_group("gloo", timeout=timeout)  # CPU DDP (local testing)
        return torch.device("cpu"), local
    if torch.cuda.is_available():
        return torch.device("cuda", 0), 0
    return torch.device("cpu"), 0


def barrier():
    if is_dist():
        dist.barrier()


def all_reduce_mean(t):
    if is_dist():
        t = t.clone()
        dist.all_reduce(t)
        t /= world()
    return t


def all_gather_object(obj):
    if not is_dist():
        return [obj]
    out = [None] * world()
    dist.all_gather_object(out, obj)
    return out


def seed_everything(seed):
    s = seed + rank()
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def worker_init_fn(worker_id):
    s = (torch.initial_seed() + worker_id) % 2 ** 32
    random.seed(s)
    np.random.seed(s)


def now():
    return datetime.datetime.now().isoformat(timespec="seconds")


class Logger:
    """Rank-0 logger: a human-readable train.log, a JSONL record stream and a per-epoch CSV (all appended)."""

    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.enabled = is_main()
        if self.enabled:
            os.makedirs(out_dir, exist_ok=True)
            self.txt = open(os.path.join(out_dir, "train.log"), "a", encoding="utf-8")
            self.jsonl = open(os.path.join(out_dir, "train_log.jsonl"), "a", encoding="utf-8")

    def info(self, msg):
        if self.enabled:
            line = f"[{now()}] {msg}"
            print(line, flush=True)
            self.txt.write(line + "\n")
            self.txt.flush()

    def log(self, rec):
        if self.enabled:
            rec = {"time": now(), **rec}
            self.jsonl.write(json.dumps(rec) + "\n")
            self.jsonl.flush()

    def epoch_row(self, row):
        if not self.enabled:
            return
        path = os.path.join(self.out_dir, "epoch_summary.csv")
        new = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if new:
                w.writeheader()
            w.writerow(row)

    def close(self):
        if self.enabled:
            for f in (getattr(self, "txt", None), getattr(self, "jsonl", None)):
                if f and not f.closed:
                    f.close()

    def __del__(self):
        self.close()


@contextlib.contextmanager
def eval_numerics():
    """Reported-score numerics: TF32 matmuls and cuDNN autotuning off (as in tools/test.py), restored afterwards.
    With both on, the same checkpoint scored up to 0.3 AP differently on an A100."""
    prev = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.benchmark)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.benchmark = prev


def atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def env_info():
    info = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "world_size": world(),
    }
    try:
        import timm

        info["timm"] = timm.__version__
    except ImportError:  # pragma: no cover
        pass
    try:
        info["nvidia_smi"] = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                                             "--format=csv,noheader"], capture_output=True, text=True,
                                            timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return info
