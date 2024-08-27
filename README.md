# YOLOv10のファインチューニング


公式のリポジトリからフォークして、独自のデータセットでファインチューニングを行うためのリポジトリです。

<p align="center">
  <img src="figures/latency.svg" width=48%>
  <img src="figures/params.svg" width=48%> <br>
  Comparisons with others in terms of latency-accuracy (left) and size-accuracy (right) trade-offs.
</p>

[YOLOv10: Real-Time End-to-End Object Detection](https://arxiv.org/abs/2405.14458).\
Ao Wang, Hui Chen, Lihao Liu, Kai Chen, Zijia Lin, Jungong Han, and Guiguang Ding

## Performance
COCO

| Model | Test Size | #Params | FLOPs | AP<sup>val</sup> | Latency |
|:---------------|:----:|:---:|:--:|:--:|:--:|
| [YOLOv10-N](https://huggingface.co/jameslahm/yolov10n) |   640  |     2.3M    |   6.7G   |     38.5%     | 1.84ms |
| [YOLOv10-S](https://huggingface.co/jameslahm/yolov10s) |   640  |     7.2M    |   21.6G  |     46.3%     | 2.49ms |
| [YOLOv10-M](https://huggingface.co/jameslahm/yolov10m) |   640  |     15.4M   |   59.1G  |     51.1%     | 4.74ms |
| [YOLOv10-B](https://huggingface.co/jameslahm/yolov10b) |   640  |     19.1M   |  92.0G |     52.5%     | 5.74ms |
| [YOLOv10-L](https://huggingface.co/jameslahm/yolov10l) |   640  |     24.4M   |  120.3G   |     53.2%     | 7.28ms |
| [YOLOv10-X](https://huggingface.co/jameslahm/yolov10x) |   640  |     29.5M    |   160.4G   |     54.4%     | 10.70ms |

## Installation

## 環境

- pyenv
- Python 3.9.13 (公式のバージョンと合わせる)
- cuda 11.8

## Setup

### 1. リポジトリをクローン

```bash
git clone git@github.com:TechC-SugarCane/train-YOLOv10.git

cd train-YOLOv10
```

### 2. Pythonの環境構築

`pyenv`を使うので、パソコンに入っていない人は[CONTRIBUTING.md](https://github.com/TechC-SugarCane/.github/blob/main/CONTRIBUTING.md#pyenv-pyenv-win-%E3%81%AE%E3%82%A4%E3%83%B3%E3%82%B9%E3%83%88%E3%83%BC%E3%83%AB)を参考にしながらインストールしてください。

```bash
pyenv install
```

### 3. 仮想環境を作成

```bash
python -m venv .venv
```

### 4. 仮想環境を有効化

```bash
# mac
source .venv/bin/activate

# windows
.venv\Scripts\activate
```

※ 環境から抜ける場合は、`deactivate`コマンドを実行してください。

### 5. 依存パッケージをインストール

```bash
# CPUで推論を行う場合
pip install -r requirements-cpu.txt

# GPUで推論を行う場合
pip install -r requirements-gpu.txt
```

リポジトリをビルド

```
# 共通
pip install -e .
```

### 6. デフォルトセッティングを変更

```bash
# datasetsのディレクトリを現在のディレクトリに変更
# デフォルトだと../datasetsが設定されている
yolo settings datasets_dir=.
```

## Training

事前学習済みモデルとして`yolov10x.pt`を使用するので、[公式GitHubのリリース](https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10x.pt)からダウンロードして`weights`ディレクトリに配置してください。

学習に使用するデータセットはRoboflowというサービスを使用して作成しています。

学習や評価に使用するデータセットは、

- [サトウキビ](https://universe.roboflow.com/hoku/sugarcane-3vhxz/dataset/11)
- [パイナップル](https://universe.roboflow.com/hoku/pineapple-thsih/dataset/7)

にありますが、手動でダウンロードするのは面倒なので`huggingface`にdatasetsをまとめてあります。

下記コマンドを実行して、datasetsをダウンロードしてください。

```bash
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install

git clone https://huggingface.co/datasets/TechC-SugarCane/yolov10-datasets

# git push時に発生するエラーを無効化
git config lfs.https://github.com/TechC-SugarCane/train-YOLOv10.git/info/lfs.locksverify false
```

学習後の結果は`runs/detect/<name(番号)>`に保存されます。

学習でよいスコアが出た場合は、`runs/detect/<name(番号)>/`にREADME.mdを作成してください。
その際は、[`runs/detect/README.md`](./runs/detect/README.md)を参考に作成してください。

```bash
# サトウキビをファインチューニングするコマンド
yolo detect train cfg='cfg/sugarcane.yaml' data=yolov10-datasets/sugarcane/data.yaml model=weights/yolov10x.pt name='yolov10x-sugarcane' epochs=300 batch=16 imgsz=640 device=0

# パイナップルをファインチューニングするコマンド
yolo detect train cfg='cfg/pineapple.yaml' data=yolov10-datasets/pineapple/data.yaml model=weights/yolov10x.pt name='yolov10x-pineapple' epochs=300 batch=16 imgsz=640 device=0
```

※ 上記を実行すると`yolov8n.pt`がダウンロードされますが、AMPというものの確認用に追加されているだけらしいので気にしなくて大丈夫です。
詳しくは[#106](https://github.com/THU-MIG/yolov10/issues/106)を参照してください。

ハイパーパラメーターは自由に調整してください。下記ファイルが`cfg/`にあります。このファイルの`Hyperparameters`の部分でハイパラ関連の設定ができます。

- サトウキビ: `sugarcane.yaml`
- パイナップル: `pineapple.yaml`

## コントリビューター向けガイドライン

コントリビューター向けのガイドラインについては、こちらの[CONTRIBUTING.md](https://github.com/TechC-SugarCane/.github/blob/main/CONTRIBUTING.md)を参照してください。

### ※ 注意

このリポジトリはforkなので、Pull Requestを送る際はこのリポジトリに対して送るようにしてください。

デフォルトだとbaseリポジトリが公式のリポジトリになっているので、注意してください。

`Comparing changes`でのドロップダウン(`base repository`)を、`TechC-SugarCane/train-YOLOv10`に変更してください。画面が遷移したら大丈夫です。

## Push to hub to 🤗

後で活用

Optionally, you can push your fine-tuned model to the [Hugging Face hub](https://huggingface.co/) as a public or private model:

```python
# let's say you have fine-tuned a model for crop detection
model.push_to_hub("<your-hf-username-or-organization/yolov10-finetuned-crop-detection")

# you can also pass `private=True` if you don't want everyone to see your model
model.push_to_hub("<your-hf-username-or-organization/yolov10-finetuned-crop-detection", private=True)
```

## Export
後で活用
```
# End-to-End ONNX
yolo export model=jameslahm/yolov10{n/s/m/b/l/x} format=onnx opset=13 simplify
# Predict with ONNX
yolo predict model=yolov10n/s/m/b/l/x.onnx

# End-to-End TensorRT
yolo export model=jameslahm/yolov10{n/s/m/b/l/x} format=engine half=True simplify opset=13 workspace=16
# or
trtexec --onnx=yolov10n/s/m/b/l/x.onnx --saveEngine=yolov10n/s/m/b/l/x.engine --fp16
# Predict with TensorRT
yolo predict model=yolov10n/s/m/b/l/x.engine
```
