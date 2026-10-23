# CARE-Tiny

[![CI](https://github.com/taxaceaee/care-tiny/actions/workflows/ci.yml/badge.svg)](https://github.com/taxaceaee/care-tiny/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.4%2B-ee4c2c)

A **10.0M-parameter, ~62 GFLOPs** detector for tiny objects in aerial images. On the **AI-TOD-v2** test set it
reaches **33.7 AP** and **20.3 APvt**, the highest very-tiny-object AP (< 8 px) reported on this benchmark, with
the fewest parameters and FLOPs among the top published results. It is not the best overall AP: the highest
published AP is 37.3 (VGTok, a 376M-parameter ViT-L detector at 2048 x 2048 input), see the comparison below.

## Results on AI-TOD-v2 test

CARE-Tiny: single model, single scale (800 x 800), single pass, no test-time augmentation. AP uses the AI-TOD
protocol (maxDets 1500; very tiny < 8 px, tiny 8-16 px, small 16-32 px, medium 32-64 px).

### Comparison with the best published results

The highest AI-TOD-v2 test results we found (as of October 2026), sorted by AP. Numbers of other methods are copied
from their papers; "–" = not reported. Bold = best in the column.

| Method | Venue | Backbone | Input | Params | GFLOPs | AP | AP50 | AP75 | APvt | APt | APs | APm |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| VGTok (100% tokens) [[1]](#ref1) | arXiv 2026 | Co-DETR, EVA-02 ViT-L | 2048² | 376M | – | **37.3** | **76.5** | – | 19.5 | **37.5** | **43.6** | **56.0** |
| ScaleBridge-Det (3B) [[2]](#ref2) | arXiv 2025 | multi-expert (Co-DETR) | 1024² | 3B | – | 35.7 | 72.1 | – | 16.2 | 34.5 | 39.5 | 51.4 |
| Dome-DETR-L [[3]](#ref3) | ACM MM 2025 | HGNetv2-B4 | 800² | 36.0M | 358.7 | 34.6 | 69.2 | **32.0** | 19.0 | 35.6 | 39.8 | 47.3 |
| Dome-DETR-M [[3]](#ref3) | ACM MM 2025 | HGNetv2-B2 | 800² | 23.9M | 252.6 | 34.0 | 68.4 | 29.9 | 18.4 | 34.3 | 39.0 | 46.9 |
| **CARE-Tiny**, best test epoch (115) | this repo | HGNetv2-B2 | 800² | **10.0M** | **61.7** | 33.7 | 67.6 | 29.7 | **20.3** | 34.5 | 38.1 | 45.8 |
| Dome-DETR-S [[3]](#ref3) | ACM MM 2025 | HGNetv2-B0 | 800² | 13.2M | 154.2 | 33.3 | 67.5 | 28.9 | 17.8 | 33.0 | 38.4 | 46.4 |
| **CARE-Tiny**, last epoch (160) | this repo | HGNetv2-B2 | 800² | **10.0M** | **61.7** | 33.2 | 66.8 | 28.9 | 20.2 | 34.0 | 37.3 | 45.3 |
| D³R-DETR [[4]](#ref4) | arXiv 2026 | HGNetv2-B0 | – | – | – | 31.3 | 65.1 | 26.2 | 16.6 | 30.8 | 36.8 | 44.7 |
| DQ-DETR [[5]](#ref5) | ECCV 2024 | ResNet-50 | 800² | 58.7M | 1805.4\* | 30.2 | 68.6 | 22.3 | 15.3 | 30.5 | 36.5 | 44.6 |

**Where CARE-Tiny stands.**

- **AP:** 5th of the published results. The two best methods use far larger models and inputs:
  - VGTok has 38x the parameters and reads the 800 x 800 images at 2048 x 2048 (6.6x the pixels);
  - ScaleBridge-Det has ~300x the parameters, at 1024 x 1024.
- **800 x 800 input:** 3rd, behind Dome-DETR-L (-0.9 AP) and Dome-DETR-M (-0.3 AP).
  - Against Dome-DETR-L: 3.6x fewer parameters and 5.8x fewer FLOPs.
  - Against Dome-DETR-M: 2.4x fewer parameters and 4.1x fewer FLOPs.
  - Ahead of Dome-DETR-S (+0.4 AP), which is larger (13.2M, 154 GFLOPs).
- **APvt (objects under 8 px):** 20.3, the highest reported, ahead of VGTok (19.5) and Dome-DETR-L (19.0).

**Protocols differ, so read the ranking with care.**

- **Dome-DETR and CARE-Tiny:**
  - train on trainval and score the EMA on test after every epoch, reporting the best epoch;
  - CARE-Tiny's last-epoch result, the protocol of the official AI-TOD-v2 baselines, is also given above.
- **ScaleBridge-Det:** trains at 1024 x 1024. Its paper does not state the final training split. Its experts are
  pre-trained on additional datasets (COCO and others).
- **VGTok:** its paper reports the AI-TOD-v2 rows as zero-shot from a dense checkpoint, at 2048 x 2048, with single
  scale and no TTA.
- **Peer review:** VGTok, ScaleBridge-Det and D³R-DETR are arXiv preprints, not peer-reviewed as of October 2026.
- **\*** DQ-DETR GFLOPs are averaged over the split, as reported by [[3]](#ref3).

**More CARE-Tiny details.**

- **Per class (epoch 115):** airplane 47.2, bridge 24.6, storage-tank 45.1, ship 55.1, swimming-pool 31.5,
  vehicle 35.9, person 19.9, wind-mill 10.4. AR1500 48.2, oLRP 69.5.
- **Best epoch:** epochs 101-121 all score 33.5-33.7, so epoch 115 is not an outlier.
- **GFLOPs:** measured at 800 x 800 with K = 256 refined queries; 60.9 at K = 128 and 66.4 at K = 1024. K is
  chosen per image from the estimated object count.
- **Logs:** every test evaluation, the loss curves, the configuration and the environment are in
  [results/aitodv2_b2_seed0](results/aitodv2_b2_seed0).

<a id="ref1"></a>[1] K. Islam. *Variable-Granularity Tokenization for High-Resolution Object Detection.*
arXiv:2608.28706, 2026. https://arxiv.org/abs/2608.28706

<a id="ref2"></a>[2] Z. Zhao, Y. Huang, L. Sun, C. Li, J. Tang. *Bridging the Scale Gap: Balanced Tiny and General
Object Detection in Remote Sensing Imagery.* arXiv:2512.01665, 2025. https://arxiv.org/abs/2512.01665

<a id="ref3"></a>[3] *Dome-DETR: DETR with Density-Oriented Feature-Query Manipulation for Efficient Tiny Object
Detection.* ACM MM 2025. https://arxiv.org/abs/2505.05741

<a id="ref4"></a>[4] *D³R-DETR: DETR with Dual-Domain Density Refinement for Tiny Object Detection in Aerial
Images.* arXiv:2601.02747, 2026. https://arxiv.org/abs/2601.02747

<a id="ref5"></a>[5] *DQ-DETR: DETR with Dynamic Query for Tiny Object Detection.* ECCV 2024.
https://arxiv.org/abs/2404.03507

## How it works

```mermaid
flowchart LR
    I[image] --> B[HGNetv2-B2 backbone]
    B --> N[FPN + PAN neck]
    N --> H[dense head<br/>stride-2 sub-slots]
    H --> R[candidate routing<br/>M = 4096, K adaptive]
    B --> M[shared maps:<br/>stride-2 detail, stride-1 pixels,<br/>PAN semantics]
    R --> R1[readout 1<br/>16x16-cell window]
    M --> R1
    R1 --> R2[readout 2<br/>re-centred window + KNN attention]
    M --> R2
    R2 --> P[refined + coarse boxes -> NMS]
```

- **Dense periphery, sparse refinement.** A cheap dense head on a stride-2 sub-slot grid finds candidates; only K of
  them (128-2048, from the estimated object count) are refined, so the expensive part scales with the number of
  objects, not the number of pixels.
- **Pixel-grounded boxes.** Each readout samples a 16 x 16-cell window from shared maps (stride-2 detail, stride-1
  pixels, PAN semantics) and predicts every box edge as a distribution over the window's cell boundaries, so a
  2-8 px object is localised on the pixel grid.
- **Cascade.** The second readout re-centres its window on the first readout's box, and its queries first attend to
  their 16 nearest neighbours (one-to-one assignment, duplicates suppressed before NMS).
- **Training.** Task-aligned assignment with NWD for small objects, quality focal loss with an IoU / NWD target,
  denoising queries, repeat-factor sampling and copy-paste of rare classes, EMA, flat-cosine schedule.

## Installation

```bash
git clone https://github.com/taxaceaee/care-tiny.git
cd care-tiny
pip install -e .            # or: pip install -r requirements.txt
```

Requires Python 3.10+, PyTorch 2.4+ and timm 1.0.9+ (the HGNetv2-B2 ImageNet weights are downloaded by timm on the
first training run).

## Data

Download [AI-TOD-v2](https://chasel-tsui.github.io/AI-TOD-v2/) and arrange it in the official layout:

```
aitodv2/
  annotations/aitodv2_{train,val,trainval,test}.json
  images/trainval/*.png
  images/test/*.png
```

Check the copy against the official release (image / box counts, SHA-256 of the annotation files, split
consistency; `--full` also decodes every image):

```bash
python tools/verify_dataset.py --root /data/aitodv2
```

## Usage

**Pretrained weights** (EMA of epoch 115, 33.7 AP, 38.7 MiB) from the
[v1.0.0 release](https://github.com/taxaceaee/care-tiny/releases/tag/v1.0.0):

```bash
mkdir -p weights
curl -L -o weights/care_tiny_b2_aitodv2_ep115.pth \
    https://github.com/taxaceaee/care-tiny/releases/download/v1.0.0/care_tiny_b2_aitodv2_ep115.pth
# sha256: ff7286e7e47d0b18d78c61b66db0ce67e53dd5bf678963330e8ce97a0412161d
```

The file holds only tensors and plain containers and loads with `torch.load(..., weights_only=True)`.

**Evaluate** a checkpoint on the test split (writes all metrics and per-class AP):

```bash
python tools/test.py --checkpoint weights/care_tiny_b2_aitodv2_ep115.pth --data-root /data/aitodv2
```

**Detect** objects in your own images (JSON + drawings):

```bash
python tools/infer.py --checkpoint weights/care_tiny_b2_aitodv2_ep115.pth --input path/to/images --score-thr 0.3
```

**Train** (one GPU; the released model took ~16.5 h on an RTX PRO 6000 including 60 test evaluations, ~24 GB of
GPU memory at batch 16; re-running the same command resumes from `checkpoints/last.pth`):

```bash
python tools/train.py --config configs/care_tiny_b2_aitodv2.yaml --data-root /data/aitodv2 --output-dir runs/b2
torchrun --nproc_per_node=2 tools/train.py --config configs/care_tiny_b2_aitodv2.yaml --data-root /data/aitodv2 \
    --output-dir runs/b2
```

To train on train, select on val and test once (no test-set selection), add `configs/protocol_train_val.yaml`.
Any key can be overridden, e.g. `--opts train.batch_size=8 data.num_workers=4`. GPUs without bf16 (e.g. T4, V100)
need `--opts train.amp_dtype=fp16`.

**Benchmark** parameters, GFLOPs and latency: `python tools/benchmark.py --k 128 256 1024`.

## Reproducibility

- `configs/care_tiny_b2_aitodv2.yaml` is the exact recipe of the released model (seed 0, batch 16, 160 epochs =
  120 with strong augmentation + 40 without). This code was checked to reproduce the research code that trained
  it bit for bit on CPU: parameters and initialisation, forward outputs, losses, gradients, augmentations, sampler
  order, a short end-to-end training run with the test protocol, and the 170 keys of the run's
  `config_resolved.json` (which uses the research code's key names).
- **Known issue, kept for reproducibility:** the class weights of the loss use the box counts in
  `criterion.AITOD_BOX_COUNTS_LEGACY`, which are not the AI-TOD-v2 trainval counts (e.g. airplane 14 325 instead of
  843). `--opts loss.class_counts=data` uses the counts of the training split instead; it has not been trained at the
  full schedule.

## Repository layout

```
care_tiny/      model, losses, label assignment, data + augmentation, training engine, AI-TOD evaluator
configs/        released recipe and the train/val protocol overlay
tools/          train.py, test.py, infer.py, benchmark.py, verify_dataset.py
tests/          unit and end-to-end tests (pytest, CPU, ~1 min)
results/        logs of the released run
```

Run the tests with `pip install -e ".[dev]" && pytest`.

## Acknowledgements

The evaluation follows [cocoapi-aitod](https://github.com/jwwangchn/cocoapi-aitod) (and is unit-tested against
pycocotools); the test protocol follows [Dome-DETR](https://github.com/RicePasteM/Dome-DETR) and
[D-FINE](https://github.com/Peterande/D-FINE); the backbone is HGNetv2 from [timm](https://github.com/huggingface/pytorch-image-models).
Dataset: [AI-TOD-v2](https://chasel-tsui.github.io/AI-TOD-v2/).

## Citation

```bibtex
@software{care_tiny_2026,
  title  = {CARE-Tiny: a 10M-parameter detector for tiny objects in aerial images},
  year   = {2026},
  url    = {https://github.com/taxaceaee/care-tiny}
}
```

## License

[Apache-2.0](LICENSE). The AI-TOD-v2 dataset is not part of this repository and has its own terms.
