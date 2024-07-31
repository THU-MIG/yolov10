# カスタムデータセット

Roboflowというサービスを使用してデーターセットを作成しています。

学習や評価に使用するデータセットは、

- [サトウキビ](https://universe.roboflow.com/hoku/sugarcane-3vhxz/dataset/11)
- [パイナップル](https://universe.roboflow.com/hoku/pineapple-thsih/dataset/7)

から`YOLO v8`形式でダウンロードし、下記を参考に`train`, `valid`, `test`を各ディレクトリに配置してください。

## ディレクトリ構造

```plaintext
datasets/
    .gitignore
    README.md
    sugarcane/
        data.yaml
        README.dataset.txt
        README.roboflow.txt
        train/
            images/
                ...
            labels/
                ...
        valid/
            images/
                ...
            labels/
                ...
        test/
            images/
                ...
            labels/
                ...
    pineapple/
        data.yaml
        README.dataset.txt
        README.roboflow.txt
        train/
            images/
                ...
            labels/
                ...
        valid/
            images/
                ...
            labels/
                ...
        test/
            images/
                ...
            labels/
                ...
```
