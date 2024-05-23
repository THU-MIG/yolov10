from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import ops
import torch

class YOLOv10DetectionValidator(DetectionValidator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.args.save_json |= self.is_coco

    def postprocess(self, preds):
        if self.training:
            preds = preds["one2one"]

        if not isinstance(preds, (list, tuple)):
            preds = [preds, None]
        
        prediction = preds[0].transpose(-1, -2)
        _, _, nd = prediction.shape
        nc = nd - 4
        assert(self.nc == nc)
        bboxes, scores = prediction.split((4, nd-4), dim=-1)
        bboxes = ops.xywh2xyxy(bboxes)

        scores, index = torch.topk(scores.flatten(1), self.args.max_det, axis=-1)
        labels = index % self.nc
        index = torch.div(index, self.nc, rounding_mode='floor')
        bboxes = bboxes.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, bboxes.shape[-1]))
        
        return torch.cat([bboxes, scores.unsqueeze(-1), labels.unsqueeze(-1)], dim=-1)