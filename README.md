
# PyTorch Benchmark Template

このリポジトリは、PyTorchを用いた深層学習モデルのベンチマーク実験を効率的に行うためのテンプレートです。
実験設定をYAMLファイルで一元管理し、MLflowやTensorBoardによるロギングと可視化にも対応しています。

本テンプレートは、**独自のモデル・損失関数・評価指標などを柔軟に拡張可能な構成**となっており、研究開発における比較実験の土台として利用できます。

---

## 特徴

* 実験設定をYAML形式で管理
* CIFAR-10などの標準データセットに対応
* 学習・評価・ロギングの処理を統合
* MLflowおよびTensorBoardによる結果の可視化
* Docker対応により再現性の高い実行環境を構築可能

---

## ディレクトリ構成

```
pytorch-benchmark-template/
├── configs/                 実験設定ファイル（YAML形式）
├── data/                    データセットの定義スクリプト
├── loggings/                ログ出力用ディレクトリ
├── models/                  モデル定義ファイル
├── trainer/                 学習・評価ロジック
├── utils/                   各種補助関数（設定管理・可視化など）
├── main.py                  実行エントリーポイント
├── schedule_run.sh          複数実験の一括実行スクリプト
├── schedule.txt             実験スケジュール定義ファイル
├── mlflow.sh                MLflow UI 起動用スクリプト
├── tensorboard.sh           TensorBoard 起動用スクリプト
├── Dockerfile               Docker環境構築用ファイル
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

```bash
git clone https://github.com/KazukiKanaya-0106/pytorch-benchmark-template.git
cd pytorch-benchmark-template

python -m venv .venv
source .venv/bin/activate
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

* `--base`（略記: `-b`）
  共通設定ファイル（任意）。指定しない場合は `configs/base.yml` が使用されます。

### 実行例

```bash
python main.py -c configs/example.yml -k demo
```

---

## 設定ファイルの例（YAML）

```yaml
data:
  dataset: CIFAR10
  data_frac: 1.0
  split:
    train: 0.7
    validation: 0.3

training:
  epochs: 50
  model: mlp
  loss: cross_entropy
  optimizer: adam

evaluation:
  metrics: [accuracy]
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
```

標準では `http://localhost:5000` にアクセスします。

### TensorBoard の起動

```bash
bash tensorboard.sh
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
```

これは次のコマンドと等価です：

```bash
python main.py -c configs/example.yml -k demo
```

複数行を記載すれば、順番に実行されます。

---

## 想定される活用例

* 異なるモデル（MLP, CNNなど）の比較実験
* 最適化手法（Adam, SGDなど）の性能比較
* 正則化やデータ拡張の有無による影響の可視化
* オリジナル損失関数・アーキテクチャの性能検証
* 異常検知や生成モデルタスクへの応用

---

## カスタマイズについて

このテンプレートは、個々の目的に応じて以下のようなカスタマイズを前提としています。

* 新しいモデルの追加（`models/`）
* データセットや前処理の追加（`data/`）
* 損失関数・ロガーの変更など

独自のベンチマークフレームワークを構築する際の出発点として活用してください。

---

## ライセンス

MIT License

---

## 作者

Kazuki Kanaya
GitHub: [@KazukiKanaya-0106](https://github.com/KazukiKanaya-0106)
