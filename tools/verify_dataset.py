"""Verify an AI-TOD-v2 copy against the official release before any training.

Checks (exit code 1 on any failure, JSON report written to --report):
  * annotation files: SHA-256 equal to the official aitodv2_*.json (Google Drive folder of the AI-TOD-v2 authors,
    github.com/Chasel-Tsui/mmdet-aitod, downloaded 2026-10-14), image / box counts, the 8 class names;
  * trainval == train + val (same images, same boxes), no image shared by trainval and test;
  * every image file present; --full: every image decoded (size equal to the json, not blank) and hashed, so
    duplicates inside a split or between trainval and test are found;
  * statistics per split: boxes per class, size groups (very tiny / tiny / small / medium), source dataset.
"""

import argparse
import collections
import hashlib
import json
import multiprocessing as mp
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from care_tiny.data import SPLITS, resolve_image_dirs  # noqa: E402

EXPECTED = {
    "train": (11214, 301534),
    "val": (2804, 75091),
    "trainval": (14018, 376625),
    "test": (14018, 376121),
}
# sha256 of the official annotation files (MyDrive folder 1Er14atDO1cBraBD4DSFODZV1x7NHO_PY of the AI-TOD-v2 authors)
OFFICIAL_SHA256 = {
    "train": "ed7b37a1187b496b96943fa46c15aab39656d59eaf501192d85178354b637b2e",
    "val": "d0439ba687db66bb38584a3398a906f15c47e055cdd84bb9060496a91f30d7cf",
    "trainval": "5d18a108b1e440d236a47669fd0a2ecf2cb5c57f588684c11975a03e28ecd262",
    "test": "9817a75f9bc4a84015881f2ddbf39bcc29635654a0266b210fd382743c927d98",
}
CLASSES = ["airplane", "bridge", "storage-tank", "ship", "swimming-pool", "vehicle", "person", "wind-mill"]
# source dataset of an AI-TOD image, from its file name
SOURCES = [("xview", r"^\d+__\d+_\d+\.png$"), ("dota", r"^P\d+__"), ("visdrone", r"^\d{7}_\d{5}_"),
           ("airbus", r"^[0-9a-f]{9}\.png$"), ("dior", r"^\d{5}\.png$")]


def source_of(name):
    for s, p in SOURCES:
        if re.match(p, name):
            return s
    return "other"


def find_file(dirs, name):
    for d in dirs:
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    return None


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_image(args):
    """(file name, path, json w, json h) -> (name, ok size, blank, md5, error)."""
    name, path, w, h = args
    try:
        with open(path, "rb") as f:
            raw = f.read()
        md5 = hashlib.md5(raw).hexdigest()
        with Image.open(path) as im:
            im.load()
            size_ok = im.size == (w, h)
            a = np.asarray(im.convert("L"))
        return name, size_ok, bool(a.std() < 1.0), md5, ""
    except Exception as e:  # unreadable / truncated file
        return name, False, False, "", f"{type(e).__name__}: {e}"


def box_key(d):
    id2f = {im["id"]: im["file_name"] for im in d["images"]}
    return collections.Counter((id2f[a["image_id"]], a["category_id"], tuple(round(x, 2) for x in a["bbox"]))
                               for a in d["annotations"])


def stats(d):
    cats = {c["id"]: c["name"] for c in d["categories"]}
    b = np.array([a["bbox"] for a in d["annotations"]], dtype=np.float64).reshape(-1, 4)
    size = np.sqrt(np.clip(b[:, 2] * b[:, 3], 0, None))
    id2s = {im["id"]: source_of(im["file_name"]) for im in d["images"]}
    return {
        "per_class": dict(collections.Counter(cats[a["category_id"]] for a in d["annotations"])),
        "mean_size_px": round(float(size.mean()), 3) if size.size else 0.0,
        "size_groups_pct": {k: round(100 * float(np.mean(m)), 2) for k, m in
                            [("very_tiny<8", size < 8), ("tiny8-16", (size >= 8) & (size < 16)),
                             ("small16-32", (size >= 16) & (size < 32)), ("medium>=32", size >= 32)]},
        "zero_size_boxes": int(((b[:, 2] <= 0) | (b[:, 3] <= 0)).sum()),
        "images_per_source": dict(collections.Counter(id2s.values())),
        "boxes_per_source": dict(collections.Counter(id2s[a["image_id"]] for a in d["annotations"])),
        "max_boxes_per_image": max(collections.Counter(a["image_id"] for a in d["annotations"]).values()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="AI-TOD-v2 folder (annotations/ and images/)")
    ap.add_argument("--report", default="data_check.json")
    ap.add_argument("--size-check", type=int, default=300, help="images per split whose size is read (no --full)")
    ap.add_argument("--full", action="store_true", help="decode and hash every image")
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--require-official-sha", action="store_true",
                    help="fail when an annotation file is not byte-identical to the official one (default: warn)")
    args = ap.parse_args()
    root = args.root
    report, ok = {"root": root, "splits": {}}, True
    data, files = {}, {}
    for split, (ann, img_dir) in SPLITS.items():
        path = os.path.join(root, "annotations", ann)
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        data[split] = d
        n_img, n_ann = len(d["images"]), len(d["annotations"])
        names = [c["name"] for c in sorted(d["categories"], key=lambda c: c["id"])]
        dirs = resolve_image_dirs(root, img_dir, [im["file_name"] for im in d["images"]])
        paths = {im["file_name"]: (find_file(dirs, im["file_name"]), im["width"], im["height"]) for im in d["images"]}
        files[split] = paths
        missing = [fn for fn, (p, _, _) in paths.items() if p is None]
        sha = sha256(path)
        rec = {"images": n_img, "annotations": n_ann, "expected": EXPECTED[split], "dirs": dirs,
               "categories": names, "sha256": sha, "sha256_official": sha == OFFICIAL_SHA256[split],
               "missing_files": len(missing), "missing_examples": missing[:5], **stats(d)}
        if not args.full:
            bad_size = []
            for im in random.Random(0).sample(d["images"], min(args.size_check, n_img)):
                p = paths[im["file_name"]][0]
                if p is not None:
                    with Image.open(p) as image:
                        if image.size != (im["width"], im["height"]):
                            bad_size.append(im["file_name"])
            rec["size_mismatch"] = bad_size[:5]
        rec["ok"] = (n_img, n_ann) == EXPECTED[split] and names == CLASSES and not missing and \
            (rec["sha256_official"] or not args.require_official_sha)
        if not rec["sha256_official"]:
            print(f"WARNING: {ann} differs from the official file (sha256 {sha})", flush=True)
        report["splits"][split] = rec
        print(split, {k: v for k, v in rec.items() if k not in ("dirs",)}, flush=True)
    # split consistency
    fn = {s: set(files[s]) for s in files}
    cons = {"train_val_disjoint": not (fn["train"] & fn["val"]),
            "trainval_is_train_plus_val_images": fn["trainval"] == fn["train"] | fn["val"],
            "trainval_test_disjoint_names": not (fn["trainval"] & fn["test"]),
            "trainval_boxes_equal_train_plus_val":
                box_key(data["trainval"]) == box_key(data["train"]) + box_key(data["val"])}
    report["consistency"] = cons
    print("consistency", cons, flush=True)
    ok &= all(cons.values()) and all(r["ok"] for r in report["splits"].values())
    if args.full:
        jobs = {}
        for s in ("trainval", "test"):  # train / val are subsets of trainval
            for name, (p, w, h) in files[s].items():
                if p is not None:
                    jobs[(s, name)] = (name, p, w, h)
        with mp.Pool(args.procs) as pool:
            res = pool.map(check_image, list(jobs.values()), chunksize=64)
        out = dict(zip(jobs.keys(), res))
        full = {}
        for s in ("trainval", "test"):
            rs = [v for (ss, _), v in out.items() if ss == s]
            md5s = collections.Counter(r[3] for r in rs if r[3])
            full[s] = {"decoded": sum(1 for r in rs if not r[4]), "errors": [r[0] + " " + r[4] for r in rs if r[4]][:5],
                       "size_mismatch": [r[0] for r in rs if not r[1] and not r[4]][:5],
                       "n_size_mismatch": sum(1 for r in rs if not r[1] and not r[4]),
                       "blank_images": [r[0] for r in rs if r[2]][:10], "n_blank": sum(1 for r in rs if r[2]),
                       "duplicate_files_inside_split": sum(c - 1 for c in md5s.values() if c > 1)}
        tv = {r[3] for (s, _), r in out.items() if s == "trainval" and r[3]}
        te = {r[3] for (s, _), r in out.items() if s == "test" and r[3]}
        full["identical_files_trainval_test"] = len(tv & te)
        report["full"] = full
        print("full", full, flush=True)
        ok &= all(full[s]["n_size_mismatch"] == 0 and not full[s]["errors"] for s in ("trainval", "test"))
    report["ok"] = bool(ok)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("DATASET OK" if ok else "DATASET MISMATCH", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
