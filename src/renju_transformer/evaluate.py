"""Evaluation helpers for training."""

from __future__ import annotations

import sys

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    desc: str | None = None,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    progress = (
        tqdm(dataloader, desc=desc, leave=True, dynamic_ncols=True, file=sys.stdout)
        if desc
        else dataloader
    )

    for input_ids, labels in progress:
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        logits = model(input_ids)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=-1) == labels).sum().item()
        total_samples += batch_size

        if desc:
            progress.set_postfix(
                loss=f"{total_loss / total_samples:.4f}",
                acc=f"{total_correct / total_samples:.4f}",
            )

    if desc:
        progress.close()

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }
