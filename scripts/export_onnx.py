#!/usr/bin/env python3
r"""Export a trained RenjuTransformer checkpoint to ONNX plus browser metadata.

Usage:
    uv run python .\scripts\export_onnx.py ^
        --checkpoint artifacts/checkpoints/best_model.pt ^
        --output web/renju_transformer.onnx
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from renju_transformer.model import RenjuTransformerModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to a saved PyTorch checkpoint.")
    parser.add_argument("--output", required=True, help="Path to write the ONNX model.")
    parser.add_argument(
        "--metadata-output",
        default=None,
        help="Optional path to write metadata JSON. Defaults to <output>.metadata.json.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=18,
        help="ONNX opset version to target. Defaults to 18.",
    )
    return parser.parse_args()


def build_model_from_checkpoint(checkpoint: dict) -> tuple[RenjuTransformerModel, dict]:
    checkpoint_config = checkpoint.get("config")
    if checkpoint_config is None:
        raise ValueError("Checkpoint does not contain embedded config data.")

    model_cfg = checkpoint_config["model"]
    model = RenjuTransformerModel(
        vocab_size=model_cfg["token_vocab_size"],
        max_seq_len=model_cfg["max_seq_len"],
        d_model=model_cfg["d_model"],
        nhead=model_cfg["nhead"],
        num_layers=model_cfg["num_layers"],
        dim_feedforward=model_cfg["dim_feedforward"],
        dropout=model_cfg["dropout"],
        activation=model_cfg["activation"],
        norm_first=model_cfg["norm_first"],
        num_move_labels=model_cfg["num_move_labels"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint_config


def build_sample_input(checkpoint_config: dict) -> torch.Tensor:
    data_cfg = checkpoint_config.get("data", {})
    model_cfg = checkpoint_config["model"]
    seq_len = int(model_cfg["max_seq_len"])
    sep_token_id = int(data_cfg.get("sep_token_id", 228))

    sample = torch.zeros((1, seq_len), dtype=torch.long)
    sample[0, -1] = sep_token_id
    return sample


def build_metadata(checkpoint_config: dict) -> dict:
    data_cfg = checkpoint_config.get("data", {})
    model_cfg = checkpoint_config["model"]
    board_size = int(data_cfg.get("board_size", 15))
    board_cells = board_size * board_size

    return {
        "model_type": "RenjuTransformerModel",
        "input_name": "input_ids",
        "output_name": "logits",
        "input_dtype": "int64",
        "output_dtype": "float32",
        "board_size": board_size,
        "board_cells": board_cells,
        "input_length": int(model_cfg["max_seq_len"]),
        "sep_token_id": int(data_cfg.get("sep_token_id", 228)),
        "move_id_offset": int(data_cfg.get("move_id_offset", 3)),
        "num_move_labels": int(model_cfg["num_move_labels"]),
        "supports_batch": False,
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    checkpoint_path = Path(args.checkpoint).resolve()
    output_path = Path(args.output).resolve()
    metadata_path = (
        Path(args.metadata_output).resolve()
        if args.metadata_output
        else output_path.with_suffix(f"{output_path.suffix}.metadata.json")
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model, checkpoint_config = build_model_from_checkpoint(checkpoint)
    sample_input = build_sample_input(checkpoint_config)

    try:
        import onnx  # noqa: F401
        import onnxscript  # noqa: F401
    except ModuleNotFoundError as exc:
        missing = exc.name or "onnx dependency"
        raise SystemExit(
            "ONNX export dependencies are missing. "
            f"Install `{missing}` and retry. "
            "For this project, run either "
            "`uv add onnx onnxscript` to add them permanently, or "
            "`uv run --with onnx --with onnxscript python .\\scripts\\export_onnx.py ...` "
            "to use an isolated ephemeral environment."
        ) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx_program = torch.onnx.export(
        model,
        (sample_input,),
        opset_version=args.opset,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamo=True,
        external_data=False,
        dynamic_shapes=None,
        optimize=True,
        verify=False,
    )
    onnx_program.save(str(output_path))

    metadata = build_metadata(checkpoint_config)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"onnx_model={output_path}")
    print(f"metadata_json={metadata_path}")


if __name__ == "__main__":
    main()
