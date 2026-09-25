"""The ReDSM5 BERT baseline, kept identical to the authors' notebook, plus seeds and saved outputs."""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertForSequenceClassification, BertTokenizer

from .data import binarizer, check_against_paper, load, split


def pick_device(requested=None):
    """CUDA, else Apple GPU, else CPU. The recipe is unchanged; only where it runs differs."""
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


class TextDataset(Dataset):
    """Tokenise everything up front and pad to the longest in the split, like the notebook."""

    def __init__(self, texts, labels, tokenizer, max_len):
        self.enc = tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        self.labels = torch.tensor(labels, dtype=torch.float)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {k: v[i] for k, v in self.enc.items()}, self.labels[i]


@torch.no_grad()
def predict_probs(model, loader, device):
    model.eval()
    out = []
    for inputs, _ in loader:
        inputs = {k: v.to(device) for k, v in inputs.items()}
        out.append(torch.sigmoid(model(**inputs).logits).cpu().numpy())
    return np.concatenate(out)


def metrics(y_true, probs, classes, threshold):
    preds = (probs > threshold).astype(int)
    rep = classification_report(y_true, preds, target_names=classes, zero_division=0, output_dict=True)
    return {
        "threshold": threshold,
        "micro_f1": float(f1_score(y_true, preds, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, preds, average="weighted", zero_division=0)),
        "exact_match_accuracy": float(accuracy_score(y_true, preds)),
        "per_class": {c: {"precision": rep[c]["precision"], "recall": rep[c]["recall"],
                          "f1": rep[c]["f1-score"], "support": int(rep[c]["support"])} for c in classes},
    }


def run(cfg, seed):
    set_seed(seed)
    device = pick_device(cfg.get("device"))
    out_dir = Path(cfg["out_dir"]) / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    posts, _ = load(cfg["data_dir"])
    if cfg.get("require_paper_match", True):
        check = check_against_paper(posts)
        if not check["matches_paper"]:
            raise SystemExit(f"data does not match the paper, refusing to train: {check}")
    train_df, test_df = split(posts)
    mlb = binarizer(posts)
    classes = list(mlb.classes_)
    y_train = mlb.transform(train_df["labels"])
    y_test = mlb.transform(test_df["labels"])

    tok = BertTokenizer.from_pretrained(cfg["model_name"])
    train_loader = DataLoader(TextDataset(train_df["text"].tolist(), y_train, tok, cfg["max_len"]),
                              batch_size=cfg["batch_size"], shuffle=True)
    test_loader = DataLoader(TextDataset(test_df["text"].tolist(), y_test, tok, cfg["max_len"]),
                             batch_size=cfg["batch_size"], shuffle=False)

    model = BertForSequenceClassification.from_pretrained(
        cfg["model_name"], num_labels=len(classes), problem_type="multi_label_classification").to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=cfg["lr"])

    t0 = time.time()
    model.train()
    for epoch in range(cfg["epochs"]):
        loop = tqdm(train_loader, leave=False, desc=f"seed {seed} epoch {epoch + 1}/{cfg['epochs']}")
        for inputs, targets in loop:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            targets = targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(**inputs).logits, targets)
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=f"{loss.item():.4f}")

    probs = predict_probs(model, test_loader, device)
    m = metrics(y_test, probs, classes, cfg["threshold"])
    m.update({"seed": seed, "train_seconds": round(time.time() - t0, 1), "device": str(device),
              "n_train": len(train_df), "n_test": len(test_df), "classes": classes, "config": cfg})

    (out_dir / "metrics.json").write_text(json.dumps(m, indent=2))
    pd.DataFrame(probs, columns=classes).assign(post_id=test_df["post_id"].values) \
        .to_csv(out_dir / "test_probs.csv", index=False)
    pd.DataFrame(y_test, columns=classes).assign(post_id=test_df["post_id"].values) \
        .to_csv(out_dir / "test_gold.csv", index=False)
    if cfg.get("save_checkpoint", True):
        model.save_pretrained(out_dir / "checkpoint")
        tok.save_pretrained(out_dir / "checkpoint")

    print(f"[seed {seed}] micro-F1 {m['micro_f1']:.3f}  macro-F1 {m['macro_f1']:.3f}  "
          f"weighted-F1 {m['weighted_f1']:.3f}  exact-match {m['exact_match_accuracy']:.3f}  "
          f"({m['train_seconds']}s on {device})")
    return m


def summarise(results, out_path):
    keys = ["micro_f1", "macro_f1", "weighted_f1", "exact_match_accuracy"]
    summary = {k: {"mean": float(np.mean([r[k] for r in results])),
                   "std": float(np.std([r[k] for r in results])),
                   "values": [r[k] for r in results]} for k in keys}
    summary["paper_reference"] = {"micro_f1": 0.51, "macro_f1": 0.36, "tolerance": 0.02}
    summary["within_tolerance"] = abs(summary["micro_f1"]["mean"] - 0.51) <= 0.02
    summary["per_class"] = {}
    for c in results[0]["classes"]:
        f1s = [r["per_class"][c]["f1"] for r in results]
        summary["per_class"][c] = {"f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s)),
                                   "support": results[0]["per_class"][c]["support"]}
    out_path.write_text(json.dumps(summary, indent=2))

    print("\nsummary over seeds", [r["seed"] for r in results])
    for k in keys:
        print(f"  {k:<22} {summary[k]['mean']:.3f} +/- {summary[k]['std']:.3f}")
    print(f"  paper micro-F1 0.51 / macro-F1 0.36 -> within tolerance: {summary['within_tolerance']}")
    print("  per-symptom F1 (mean):")
    for c, v in summary["per_class"].items():
        print(f"    {c:<18} {v['f1_mean']:.3f} +/- {v['f1_std']:.3f}  (n={v['support']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/bert_baseline.yaml")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    seeds = args.seeds or [args.seed if args.seed is not None else cfg.get("seed", 0)]
    results = [run(cfg, s) for s in seeds]
    # one seed has its own metrics.json; writing summary.json here would overwrite the five-seed one
    if len(results) > 1:
        summarise(results, Path(cfg["out_dir"]) / "summary.json")


if __name__ == "__main__":
    main()
