from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import ops
import torch

class YOLOv10DetectionValidator(DetectionValidator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.args.save_json |= self.is_coco

    def postprocess(self, preds):
        if isinstance(preds, dict):
            preds = preds["one2one"]

        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        
        preds = preds.transpose(-1, -2)
        boxes, scores, labels = ops.v10postprocess(preds, self.args.max_det, self.nc)
        bboxes = ops.xywh2xyxy(boxes)
        return torch.cat([bboxes, scores.unsqueeze(-1), labels.unsqueeze(-1)], dim=-1)