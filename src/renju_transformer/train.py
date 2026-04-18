"""Training pipeline for Renju next-move prediction."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader, random_split

from .dataset import RenjuDataset
from .evaluate import evaluate_model
from .model import RenjuTransformerModel
from .tokenizer import RenjuTokenizer
from .utils import ensure_mlflow_experiment, flatten_config, select_device, set_seed


def build_model(cfg: DictConfig) -> RenjuTransformerModel:
    return RenjuTransformerModel(
        vocab_size=cfg.model.token_vocab_size,
        max_seq_len=cfg.model.max_seq_len,
        d_model=cfg.model.d_model,
        nhead=cfg.model.nhead,
        num_layers=cfg.model.num_layers,
        dim_feedforward=cfg.model.dim_feedforward,
        dropout=cfg.model.dropout,
        activation=cfg.model.activation,
        norm_first=cfg.model.norm_first,
        num_move_labels=cfg.model.num_move_labels,
    )


def build_optimizer(model: nn.Module, cfg: DictConfig) -> torch.optim.Optimizer:
    if cfg.optimizer.name != "adamw":
        raise ValueError(f"Unsupported optimizer: {cfg.optimizer.name}")
    return torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
        betas=tuple(cfg.optimizer.betas),
        eps=cfg.optimizer.eps,
    )


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: DictConfig):
    if cfg.scheduler.name == "none":
        return None
    raise ValueError(f"Unsupported scheduler: {cfg.scheduler.name}")


def train_model(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    tokenizer = RenjuTokenizer(
        sep_token_id=cfg.data.sep_token_id,
        move_id_offset=cfg.data.move_id_offset,
    )
    dataset = RenjuDataset(cfg.data.path, tokenizer=tokenizer, max_rows=cfg.data.max_rows)

    train_size = int(len(dataset) * cfg.data.train_split)
    val_size = len(dataset) - train_size
    if train_size <= 0 or val_size <= 0:
        raise ValueError(
            f"Dataset split is invalid: total={len(dataset)}, "
            f"train_split={cfg.data.train_split}, train={train_size}, val={val_size}."
        )

    split_generator = torch.Generator().manual_seed(cfg.seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=split_generator)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
    )

    device = select_device(cfg.train.device)
    model = build_model(cfg).to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    criterion = nn.CrossEntropyLoss()

    output_root = Path(cfg.train.output_root)
    checkpoint_dir = output_root / cfg.train.checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config_dir = output_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    resolved_config_path = config_dir / "resolved_config.yaml"
    OmegaConf.save(cfg, resolved_config_path, resolve=True)

    ensure_mlflow_experiment(
        tracking_uri=cfg.mlflow.tracking_uri,
        experiment_name=cfg.mlflow.experiment_name,
        artifact_root=cfg.mlflow.artifact_root,
    )
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    run_name = f"{cfg.mlflow.run_name_prefix}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    best_val_loss = float("inf")
    best_checkpoint_path = checkpoint_dir / cfg.train.checkpoint_name
    best_model_state: dict[str, torch.Tensor] | None = None
    global_step = 0

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(flatten_config(cfg))
        mlflow.log_artifact(str(resolved_config_path), artifact_path="configs")

        for epoch in range(1, cfg.train.max_epochs + 1):
            model.train()
            total_loss = 0.0
            total_correct = 0
            total_samples = 0

            for step, (input_ids, labels) in enumerate(train_loader, start=1):
                input_ids = input_ids.to(device)
                labels = labels.to(device)

                optimizer.zero_grad(set_to_none=True)
                logits = model(input_ids)
                loss = criterion(logits, labels)
                loss.backward()

                if cfg.train.gradient_clip_norm is not None and cfg.train.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.gradient_clip_norm)

                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size
                total_correct += (logits.argmax(dim=-1) == labels).sum().item()
                total_samples += batch_size
                global_step += 1

                if step % cfg.train.log_every_steps == 0:
                    mlflow.log_metric("train_step_loss", loss.item(), step=global_step)

            train_metrics = {
                "loss": total_loss / total_samples,
                "accuracy": total_correct / total_samples,
            }
            val_metrics = evaluate_model(model, val_loader, criterion, device)

            mlflow.log_metric("train_loss", train_metrics["loss"], step=epoch)
            mlflow.log_metric("train_accuracy", train_metrics["accuracy"], step=epoch)
            mlflow.log_metric("val_loss", val_metrics["loss"], step=epoch)
            mlflow.log_metric("val_accuracy", val_metrics["accuracy"], step=epoch)

            print(
                f"epoch={epoch} "
                f"train_loss={train_metrics['loss']:.4f} "
                f"train_acc={train_metrics['accuracy']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_acc={val_metrics['accuracy']:.4f}"
            )

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_model_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
                checkpoint = {
                    "model_state_dict": best_model_state,
                    "config": OmegaConf.to_container(cfg, resolve=True),
                    "epoch": epoch,
                    "val_loss": best_val_loss,
                }
                torch.save(checkpoint, best_checkpoint_path)
                mlflow.log_metric("best_val_loss", best_val_loss, step=epoch)

        if best_model_state is None:
            raise RuntimeError("Training completed without producing a checkpoint.")

        mlflow.log_artifact(str(best_checkpoint_path), artifact_path="checkpoints")

        if cfg.mlflow.log_model:
            best_model = build_model(cfg)
            best_model.load_state_dict(best_model_state)
            best_model.eval()
            mlflow.pytorch.log_model(best_model, name="model")

    print(f"best_checkpoint={best_checkpoint_path.resolve()}")
