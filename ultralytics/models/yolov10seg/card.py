card_template_text = """
---
license: agpl-3.0
library_name: ultralytics
repo_url: https://github.com/THU-MIG/yolov10
tags:
- object-detection
- computer-vision
- yolov10
datasets:
- detection-datasets/coco
inference: false
---

### Model Description
[YOLOv10: Real-Time End-to-End Object Detection](https://arxiv.org/abs/2405.14458v1)

- arXiv: https://arxiv.org/abs/2405.14458v1
- github: https://github.com/THU-MIG/yolov10

### Installation
```
pip install git+https://github.com/THU-MIG/yolov10.git
```

### Training and validation
```python
from ultralytics import YOLOv10

model = YOLOv10.from_pretrained('jameslahm/yolov10n')
# Training
model.train(...)
# after training, one can push to the hub
model.push_to_hub("your-hf-username/yolov10-finetuned")

# Validation
model.val(...)
```

### Inference

Here's an end-to-end example showcasing inference on a cats image:

```python
from ultralytics import YOLOv10

model = YOLOv10.from_pretrained('jameslahm/yolov10n')
source = 'http://images.cocodataset.org/val2017/000000039769.jpg'
model.predict(source=source, save=True)
```
which shows:

![image/png](https://cdn-uploads.huggingface.co/production/uploads/628ece6054698ce61d1e7be3/tBwAsKcQA_96HCYQp7BRr.png)

### BibTeX Entry and Citation Info
```
@article{wang2024yolov10,
  title={YOLOv10: Real-Time End-to-End Object Detection},
  author={Wang, Ao and Chen, Hui and Liu, Lihao and Chen, Kai and Lin, Zijia and Han, Jungong and Ding, Guiguang},
  journal={arXiv preprint arXiv:2405.14458},
  year={2024}
}
```
""".strip()