import json
import sys

import yaml

from src import train_bert


def fake_run(cfg, seed):
    """Stands in for training: what summarise needs, with the seed as every score."""
    return {"seed": seed, "micro_f1": seed, "macro_f1": seed, "weighted_f1": seed,
            "exact_match_accuracy": seed, "classes": ["FATIGUE"],
            "per_class": {"FATIGUE": {"f1": seed, "support": 1}}}


def test_a_single_seed_does_not_overwrite_the_five_seed_summary(tmp_path, monkeypatch):
    config = tmp_path / "cfg.yaml"
    config.write_text(yaml.safe_dump({"out_dir": str(tmp_path)}))
    monkeypatch.setattr(train_bert, "run", fake_run)

    monkeypatch.setattr(sys, "argv", ["train_bert", "--config", str(config), "--seeds", "0", "1", "2", "3", "4"])
    train_bert.main()
    five = (tmp_path / "summary.json").read_text()
    assert json.loads(five)["micro_f1"]["values"] == [0, 1, 2, 3, 4]

    monkeypatch.setattr(sys, "argv", ["train_bert", "--config", str(config), "--seed", "0"])
    train_bert.main()
    assert (tmp_path / "summary.json").read_text() == five
