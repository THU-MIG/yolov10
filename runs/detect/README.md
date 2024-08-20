このREADMEは工事中です

# 学習後の保存方法について

## trainのディレクトリ構造

学習後の結果は`runs/detect/<name(番号)>`に保存されます。`<name(番号)>`は学習時にコマンドで指定したnameオプションの値です。<br>
同じnameオプションの値を指定した場合、`<name(番号)>`の値がインクリメントされます。

そのディレクトリの中身は、学習結果の可視化, モデルの重みファイル, ログファイルなどが保存されています。

## 学習後について

学習でよいスコアが出た場合は、`runs/detect/<name(番号)>/`にREADME.mdを作成してください。

その際のREADME.mdのフォーマットは以下の通りです。

コマンドに学習時のコマンドを、結果に学習後のコンソール画面のスクショパス(console.png)を記載してください。

````markdown
## コマンド

```bash
# ここに学習時のコマンドを記載してください

例:
yolo detect train \
    cfg='cfg/sugarcane.yaml' \
    data=datasets/sugarcane/data.yaml \
    model=weights/yolov10x.pt \
    name='yolov10x-sugarcane' \
    epochs=300 \
    batch=16 \
    imgsz=640 \
    device=0
```

## 学習過程

![results.png](./results.png)

## 結果

![結果のスクショを同ディレクトリ内の`console.png`に保存してください](./console.png)
````

READMEの例は<<いつか上げる。それまでは[YOLOv9の実装](https://github.com/TechC-SugarCane/train-YOLOv9/tree/main/runs/train/yolov9-e-pineapple-たたき台)を参考にしてほしい>>を参照してください。

## モデルの保存

現在GitHubに上がっているスコアより良いモデルができた場合、<<たぶんfuggingface>>に`best.pt`をアップロードしてください。
