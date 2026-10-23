# Released run: AI-TOD-v2, HGNetv2-B2, seed 0

Trained on trainval with `configs/care_tiny_b2_aitodv2.yaml`, on one NVIDIA RTX PRO 6000 Blackwell (96 GB), PyTorch
2.11 + CUDA 13.0 (`env.json`), 160 epochs in two sessions (resumed at epoch 124; see `train.log`).

| File | Content |
|---|---|
| `dfine_protocol.json` | test AP of every scored epoch (101-160), best epoch 115 (33.71), last epoch 160 (33.24), reloads |
| `eval_test_epNNN.json` | all AI-TOD metrics, per-class AP / oLRP and timing of epoch NNN on the 14 018 test images |
| `results_table.md` | best and last epoch next to Dome-DETR |
| `train.log`, `train_log.jsonl` | training log and every 50 steps: losses, LR, K, grad norm, throughput, memory |
| `epoch_summary.csv` | mean losses and time per epoch |
| `config_resolved.json`, `env.json`, `status.json` | configuration, environment, final status |

The run was produced by the research version of this code, so `config_resolved.json` uses its key names (a superset
of `care_tiny/config.py`; the release code was checked against it key by key). Paths such as `/content/runs/...`
are those of the training machine.
