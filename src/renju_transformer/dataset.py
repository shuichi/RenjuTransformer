"""Dataset loading for Renju training CSV logs."""

from __future__ import annotations

import csv
from pathlib import Path

from torch.utils.data import Dataset

from .tokenizer import RenjuTokenizer


class RenjuDataset(Dataset[tuple]):
    def __init__(self, csv_path: str | Path, tokenizer: RenjuTokenizer, max_rows: int | None = None) -> None:
        self.csv_path = Path(csv_path)
        self.tokenizer = tokenizer
        self.samples: list[tuple] = []
        self._load(max_rows=max_rows)

    def _load(self, max_rows: int | None) -> None:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.csv_path}")

        with self.csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            for row_index, raw_row in enumerate(reader, start=1):
                if not raw_row:
                    continue
                try:
                    row = [int(value) for value in raw_row]
                except ValueError as exc:
                    raise ValueError(f"Non-integer value found in row {row_index}.") from exc
                self.samples.append(self.tokenizer.encode_csv_row(row))
                if max_rows is not None and len(self.samples) >= max_rows:
                    break

        if not self.samples:
            raise ValueError(f"No training samples found in {self.csv_path}.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple:
        return self.samples[index]
