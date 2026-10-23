"""Run a checkpoint on images and save the detections (JSON) and drawings (PNG).

    python tools/infer.py --checkpoint weights/care_tiny_b2_aitodv2_ep115.pth --input path/to/image_or_folder \
        --output-dir runs/infer --score-thr 0.3

Images of any size are zero-padded to a multiple of 32; boxes are in input pixels. The detector was trained on
AI-TOD-v2 (aerial images, ground sampling distance of roughly 0.1-1 m) with the classes listed in CLASSES.
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from torchvision.io import ImageReadMode, decode_image, read_file  # noqa: E402

from care_tiny.checkpoint import load_model  # noqa: E402
from care_tiny.data import collate  # noqa: E402
from care_tiny.model import postprocess  # noqa: E402

CLASSES = ["airplane", "bridge", "storage-tank", "ship", "swimming-pool", "vehicle", "person", "wind-mill"]
COLORS = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (70, 240, 240),
          (240, 50, 230), (250, 190, 212)]
EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def list_images(path):
    if os.path.isdir(path):
        return sorted(p for p in glob.glob(os.path.join(path, "*")) if p.lower().endswith(EXTS))
    return [path]


def draw(img, det, score_thr, path):
    im = Image.fromarray(img.permute(1, 2, 0).numpy())
    d = ImageDraw.Draw(im)
    for b, s, lab in zip(det["boxes"].tolist(), det["scores"].tolist(), det["labels"].tolist()):
        if s >= score_thr:
            d.rectangle(b, outline=COLORS[lab % len(COLORS)], width=1)
    im.save(path)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--input", required=True, help="an image or a folder of images")
    ap.add_argument("--output-dir", default="runs/infer")
    ap.add_argument("--score-thr", type=float, default=0.3, help="drawing / JSON threshold (AP uses all detections)")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--opts", nargs="*", default=[])
    args = ap.parse_args()

    device = torch.device(args.device)
    model, cfg, _ = load_model(args.checkpoint, device, args.opts)
    use_amp = device.type == "cuda" and cfg["train"]["amp"]
    amp_dtype = torch.bfloat16 if cfg["train"]["amp_dtype"] == "bf16" else torch.float16
    os.makedirs(args.output_dir, exist_ok=True)
    paths = list_images(args.input)
    results = []
    for i in range(0, len(paths), args.batch_size):
        chunk = paths[i:i + args.batch_size]
        imgs = [decode_image(read_file(p), mode=ImageReadMode.RGB) for p in chunk]
        batch, _ = collate([(im, {}) for im in imgs], pad_divisor=32)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            out = model(batch.to(device))
        dets = postprocess(out, [tuple(im.shape[1:]) for im in imgs], cfg["eval"])
        for p, im, det in zip(chunk, imgs, dets):
            det = {k: v.float().cpu() if k != "labels" else v.cpu() for k, v in det.items()}
            name = os.path.splitext(os.path.basename(p))[0]
            draw(im, det, args.score_thr, os.path.join(args.output_dir, f"{name}.png"))
            keep = det["scores"] >= args.score_thr
            results.append({"image": p, "detections": [
                {"bbox_xyxy": [round(v, 2) for v in b], "score": round(s, 4), "label": CLASSES[lab]}
                for b, s, lab in zip(det["boxes"][keep].tolist(), det["scores"][keep].tolist(),
                                     det["labels"][keep].tolist())]})
            print(f"{p}: {int(keep.sum())} detections >= {args.score_thr}")
    with open(os.path.join(args.output_dir, "detections.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
