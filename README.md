# RenjuTransformer

PyTorch + Hydra + MLflow + uv で構成した、五目並べの「次の1手」予測用 TransformerEncoder プロジェクトです。

## セットアップ

```powershell
uv sync
```

## 学習

`data.csv` は `renju-mtcs.py` が出力した 1 行 227 列の CSV を想定します。

```powershell
uv run python .\renju-transformer.py
```

設定は Hydra で上書きできます。

```powershell
uv run python .\renju-transformer.py data.path=sample-log.csv train.max_epochs=5 train.batch_size=32 model.d_model=256
```

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
