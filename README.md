
# PyTorch Benchmark Template

このリポジトリは、PyTorchを用いた深層学習モデルのベンチマーク実験を効率的に行うためのテンプレートです。
CIFAR10やGLUE/SST2等のデータセットや、実験設定をYAMLファイルで一元管理し、MLflowやTensorBoardによるロギングと可視化にも対応しています。

本テンプレートは、**独自のモデル・損失関数・評価指標などを柔軟に拡張可能な構成**となっており、研究開発における比較実験の土台として利用できます。

---

## 特徴

* 実験設定をYAML形式で管理
* グリッドサーチによるハイパーパラメータ探索
* 学習・評価・ロギングの処理を統合
* MLflowおよびTensorBoardによる結果の可視化
* Docker対応により再現性の高い実行環境を構築可能

---

## ディレクトリ構成

```
pytorch-benchmark-template/
├── artifacts/               ログ出力用ディレクトリ
├── components/              主要コンポーネント作成クラス（モデル・損失関数など）
├── configs/                 実験設定ファイル（YAML形式）
├── core/                    メインエンジン作成クラス（configクラスなど）
├── data/                    データセットの定義スクリプト
├── models/                  モデル定義ディレクトリ
├── losses/                  損失関数定義ディレクトリ
├── scripts/                 学習・評価ロジック
├── utils/                   各種補助関数（ファイル操作・データ操作など）
├── main.py                  実行エントリーポイント
├── schedule_run.sh          複数実験の一括実行スクリプト
├── schedule.txt             実験スケジュール定義ファイル
├── mlflow.sh                MLflow UI 起動用スクリプト
├── tensorboard.sh           TensorBoard 起動用スクリプト
├── Dockerfile               Docker環境構築用ファイル
├── docker-compose.yml       Docker環境構築用ファイル
└── requirements.txt         依存パッケージ一覧
```

---

## セットアップ手順

### Dockerを使用する場合（推奨）

```bash
docker compose up --build
```

※ ポート番号（5000、6006など）が他プロセスと競合していないことを確認してください。

---

### ローカル環境で実行する場合


#### venv を使う場合
```bash

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

#### anaconda を使う場合
```bash

conda create -n example_env python=3.11
conda activate example_env
pip install -r requirements.txt
```

---

## 実行方法

以下のコマンドで実験を開始できます。

```bash
python main.py --config configs/example.yml
```

### オプション一覧

* `--config`（略記: `-c`）
  実験ごとの設定ファイル（必須）。例: `configs/example.yml`

* `--key`（略記: `-k`）
  実行インスタンスを識別するためのキー（任意）。ログディレクトリ名などに利用されます。

* `--grid-search`（略記: `-g`）
  グリッドサーチの対象パラメータの設定ファイル（任意）。例: `configs/grid_search/example_focal_grid.yml`

### 実行例

```bash
python main.py -c configs/example.yml -k demo
```

---

## 実験設定ファイルの例（YAML）

```yaml
training:
  dataset: cifar10
  weight: null
  epochs: 50
  model: mlp_layer7
  loss: cross_entropy_loss
  optimizer: adam

evaluation:
  metrics: [accuracy]
```

---

## グリッドサーチ設定ファイルの例（YAML）

```yaml
loss:
  binary_focal_loss:
    gamma: [1.0, 1.5, 2.0]  # 探索したいパラメータのみを配列で指定する
```

---

## ログの出力先

* モデル・設定ファイルの保存先: `artifacts/`
* TensorBoardログ: `tensorboard/`
* MLflowログ: `mlflow/`

---

## ロギングUIの起動

ログの可視化に使用するUIは以下のスクリプトで起動可能です。

### MLflow UI の起動

```bash
bash mlflow.sh

# または
mlflow ui --backend-store-uri "file:./artifacts/mlflow"
```

標準では `http://localhost:5000` にアクセスします。

### TensorBoard の起動

```bash
bash tensorboard.sh

# または
tensorboard --logdir ./artifacts/tensorboard
```

標準では `http://localhost:6006` にアクセスします。

---

## 複数実験の一括実行（スケジュール機能）

複数の設定を一括で実行したい場合は、以下のスクリプトを使用します。

### 実行方法

```bash
bash schedule_run.sh
```

### スクリプト内容（`schedule_run.sh`）

```bash
#!/bin/bash

while IFS= read -r line || [ -n "$line" ]; do
  [[ -z "$line" || "$line" =~ ^# ]] && continue
  python main.py $line || echo "[ERROR] Failed: $line" >&2
done < schedule.txt
```

### スケジュールファイルの例（`schedule.txt`）

```txt
-c configs/example.yml -k demo
-c configs/base.yml -k base
```

これは次のコマンドと等価です：

```bash
python main.py -c configs/example.yml -k demo
python main.py -c configs/base.yml -k base
```

`schedule.txt`に複数行を記載し`schedule_run.sh`を実行すれば、順番に実行されます。

---

## 想定される活用例

* 異なるモデル（MLP, CNNなど）の比較実験
* 最適化手法（Adam, SGDなど）の性能比較
* 正則化やデータ拡張の有無による影響の可視化
* オリジナルアーキテクチャの性能検証
* 異常検知や生成モデルタスクへの応用

---

## カスタマイズについて

このテンプレートは、個々の目的に応じて以下のようなカスタマイズを前提としています。

* 新しいモデルの追加（`models/`）
* 新しい損失関数の追加（`losses/`）
* データセットや前処理の追加（`data/`）

独自のベンチマークフレームワークを構築する際の出発点として活用してください。

## 作者

Kazuki Kanaya
GitHub: [@KazukiKanaya-0106](https://github.com/KazukiKanaya-0106)
