import os

import pytest
import torch

from care_tiny.config import DEFAULTS, load_config, to_dict

from conftest import ROOT

CONFIGS = os.path.join(ROOT, "configs")


def test_release_config_spells_out_the_defaults():
    assert load_config(os.path.join(CONFIGS, "care_tiny_b2_aitodv2.yaml")) == load_config(None)
    assert to_dict(load_config(None)) == DEFAULTS


def test_train_val_protocol_overlay():
    cfg = load_config([os.path.join(CONFIGS, "care_tiny_b2_aitodv2.yaml"),
                       os.path.join(CONFIGS, "protocol_train_val.yaml")])
    assert (cfg.data.train_split, cfg.data.val_split) == ("train", "val")
    assert not cfg.train.test_during_train and not cfg.train.stage2_reload and cfg.train.val_interval > 0


def test_files_merge_in_order_then_overrides(tmp_path):
    a, b = tmp_path / "a.yaml", tmp_path / "b.yaml"
    a.write_text("train:\n  batch_size: 8\n  lr: 0.0003\ndata:\n  num_workers: 4\n")
    b.write_text("data:\n  num_workers: 16\n")
    cfg = load_config([str(a), str(b)], ["train.lr=0.0002", "model.k_buckets=[16,32]"])
    assert (cfg.train.batch_size, cfg.data.num_workers, cfg.train.lr) == (8, 16, 0.0002)
    assert cfg.model.k_buckets == [16, 32] and cfg.data.aug.copy_paste_p == DEFAULTS["data"]["aug"]["copy_paste_p"]


def test_unknown_keys_are_errors(tmp_path):
    with pytest.raises(KeyError):
        load_config(None, ["train.learning_rate=1e-4"])
    bad = tmp_path / "bad.yaml"
    bad.write_text("model:\n  windw: 8\n")
    with pytest.raises(KeyError):
        load_config(str(bad))
    with pytest.raises(ValueError):
        load_config(None, ["train.lr"])


def test_checkpoint_config_is_plain_data(tmp_path):
    path = tmp_path / "c.pth"
    torch.save({"cfg": to_dict(load_config(None))}, path)
    assert torch.load(path, weights_only=True)["cfg"] == DEFAULTS
