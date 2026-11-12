"""Evaluation: accuracy (follows the digit) and shortcut rate (follows the colour)."""

import json
import os

import numpy as np
import torch
from torch import nn

# aligned: colour = true class | clean: no colour | random: colour carries no info | conflicting: colour = label + 1
TEST_SETS = ["test_aligned", "test_clean", "test_random", "test_conflicting"]


@torch.no_grad()                                      #running the whole function without computing gradients
def evaluate(model: nn.Module, loader) -> dict:
    model.eval()
    device = next(model.parameters()).device          # wherever the model is (CPU or GPU)
    correct = follows = total = 0
    for images, labels, shortcut_labels in loader:
        images, labels, shortcut_labels = images.to(device), labels.to(device), shortcut_labels.to(device)
        preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()            # follows the digit
        follows += (preds == shortcut_labels).sum().item()   # follows the colour (-1 never matches)
        total += len(labels)
    return {"accuracy": correct / total, "shortcut_rate": follows / total}


def evaluate_all(runs: dict, loaders: dict) -> dict:
    """runs: {(model_name, run_name): model} -> {"model/run/test": {...}}"""
    results = {}
    for (model_name, run_name), model in runs.items():
        for test_name in TEST_SETS:
            results[f"{model_name}/{run_name}/{test_name}"] = evaluate(model, loaders[test_name])
    return results


def save_metrics(results: dict, path: str = "results/metrics.json") -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def results_table(results: dict) -> str:
    """One row per (model, run): accuracy on each test set + shortcut rate on conflicting.
    Columns are padded so the table is readable in the terminal and is still valid markdown."""
    header = ["model", "run"] + [f"acc {t.replace('test_', '')}" for t in TEST_SETS] + ["shortcut rate (conflicting)"]
    body = []
    for model_name, run_name in sorted({tuple(key.split("/")[:2]) for key in results}):
        r = lambda test: results[f"{model_name}/{run_name}/{test}"]
        body.append([model_name, run_name]
                    + [f"{r(t)['accuracy']:.1%}" for t in TEST_SETS]
                    + [f"{r('test_conflicting')['shortcut_rate']:.1%}"])
    widths = [max(len(row[i]) for row in [header] + body) for i in range(len(header))]
    line = lambda cells: "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths)) + " |"
    separator = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    return "\n".join([line(header), separator] + [line(row) for row in body])

@torch.no_grad()
def confusion_matrix(model: nn.Module, loader) -> np.ndarray:
    """Rows = true digit, columns = prediction."""
    model.eval()
    device = next(model.parameters()).device
    cm = np.zeros((10, 10), dtype=int)
    for images, labels, _ in loader:
        preds = model(images.to(device)).argmax(dim=1).cpu()
        for t, p in zip(labels.tolist(), preds.tolist()):
            cm[t, p] += 1
    return cm


if __name__ == "__main__":
    # Quick check on REAL data: baseline vs reweighting, small and fast.
    from functools import partial

    from src.data import get_loaders
    from src.mitigations import reweighting_loss
    from src.models import build_model
    from src.train import set_seed, train

    loaders = get_loaders(n_train=5000)              # 5000 training images: quick on a CPU
    images, _, _ = next(iter(loaders["train_biased"]))

    runs = {}
    for run_name, loss_fn in [("baseline", None),
                              ("reweighting", partial(reweighting_loss, upweight=10.0))]:
        print(f"--- training {run_name}")
        set_seed(42)
        model = build_model("cnn", images)
        train(model, loaders["train_biased"], epochs=3, loss_fn=loss_fn)
        runs[("cnn", run_name)] = model

    print()
    print(results_table(evaluate_all(runs, loaders)))