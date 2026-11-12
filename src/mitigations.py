import torch
import torch.nn.functional as F


def batch_weights(labels: torch.Tensor, shortcut_labels: torch.Tensor,
                  upweight: float = 10.0) -> torch.Tensor:
    # Rare samples (shortcut does not match the label) get a larger weight
    weights = torch.ones(len(labels), device=labels.device)
    weights[shortcut_labels != labels] = upweight
    return weights


def weighted_cross_entropy(logits: torch.Tensor, labels: torch.Tensor,
                           weights: torch.Tensor) -> torch.Tensor:
    # One loss value per sample, multiplied by its weight, then averaged
    per_sample = F.cross_entropy(logits, labels, reduction="none")
    return (per_sample * weights).mean()


def reweighting_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    shortcut_labels: torch.Tensor,
    upweight: float = 10.0
) -> torch.Tensor:
    weights = batch_weights(
        labels,
        shortcut_labels,
        upweight
    )

    return weighted_cross_entropy(logits, labels, weights)
