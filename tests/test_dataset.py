from __future__ import annotations

import gzip

import torch

from renju_transformer.dataset import RenjuDataset
from renju_transformer.tokenizer import RenjuTokenizer


def test_dataset_loads_gzip_csv_into_samples(tmp_path):
    tokenizer = RenjuTokenizer()
    row = [0] * tokenizer.board_cells + [tokenizer.sep_token_id, tokenizer.move_id_offset]
    csv_path = tmp_path / "data.csv.gz"
    csv_path.write_bytes(
        gzip.compress((",".join(str(value) for value in row) + "\n").encode("utf-8"))
    )

    dataset = RenjuDataset(csv_path, tokenizer=tokenizer)

    assert len(dataset) == 1
    input_ids, label = dataset[0]
    assert torch.equal(
        input_ids, torch.tensor([0] * tokenizer.board_cells + [tokenizer.sep_token_id])
    )
    assert label.item() == 0
