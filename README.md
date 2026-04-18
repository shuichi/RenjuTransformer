# RenjuTransformer

PyTorch + Hydra + MLflow + uv で構成した、五目並べの「次の1手」予測用 TransformerEncoder プロジェクトです。

## セットアップ

```powershell
uv sync
```

## 学習

`data.csv` は `mode=generate` が出力した 1 行 227 列の CSV を想定します。

```powershell
uv run python .\renju-transformer.py
```

設定は Hydra で上書きできます。

```powershell
uv run python .\renju-transformer.py data.path=sample-log.csv train.max_epochs=5 train.batch_size=32 model.d_model=256
```

## 合成データ生成

軽量自己対戦ベースの生成パイプラインを `generate` モードで実行できます。出力形式は学習用 CSV と同じで、各行は `board(225) + SEP(228) + move_id` です。

```powershell
uv run python .\renju-transformer.py mode=generate generate.games=100 generate.output_path=synthetic-data.csv
```

デフォルトでは、基礎データ生成に加えて次の Data Augmentation を 1 回ずつ適用します。

- `generate.augmentation.move_ratio=0.10`
- `generate.augmentation.add_stones_per_color=1`
- `generate.augmentation.copies_per_sample=1`

高精度データ追加はデフォルトで off です。必要な場合のみ MCTS 生成を追加できます。

```powershell
uv run python .\renju-transformer.py mode=generate generate.high_quality.enabled=true generate.high_quality.games=20
```

Windows / macOS の両方で大量生成したい場合は、Python 側の shard 並列化を使えます。各 worker が別 CSV を生成し、最後に自動で 1 つに結合します。

```powershell
uv run python .\renju-transformer.py mode=generate generate.games=3200 generate.output_path=data.csv generate.parallel.enabled=true generate.parallel.workers=32
```

中間 shard を残したい場合は `generate.parallel.keep_shards=true` を指定します。

## 推論

学習済み checkpoint と盤面を与えると、次の一手 ID を出力します。

```powershell
uv run python .\renju-transformer.py mode=predict predict.checkpoint_path=artifacts/checkpoints/best_model.pt predict.board_path=board.txt
```

`board.txt` は 225 個の `0,1,2` をカンマ区切りで並べた 1 行ファイルです。

## MLflow

追跡 DB は SQLite、artifact はローカルディレクトリです。

```powershell
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## 設定

設定はすべて `config/` 配下の Hydra 管理です。

- `config/data/`: データセット
- `config/model/`: TransformerEncoder
- `config/train/`: 学習条件
- `config/optimizer/`: 最適化
- `config/scheduler/`: スケジューラ
- `config/mlflow/`: 実験管理
- `config/predict/`: 推論
