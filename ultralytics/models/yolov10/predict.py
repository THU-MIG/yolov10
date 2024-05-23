from ultralytics.models.yolo.detect import DetectionPredictor
import torch
from ultralytics.utils import ops
from ultralytics.engine.results import Results


class YOLOv10DetectionPredictor(DetectionPredictor):
    def postprocess(self, preds, img, orig_imgs):
        if not isinstance(preds, (list, tuple)):
            preds = [preds, None]
        
        prediction = preds[0].transpose(-1, -2)
        _, _, nd = prediction.shape
        nc = nd - 4
        bboxes, scores = prediction.split((4, nd-4), dim=-1)
        bboxes = ops.xywh2xyxy(bboxes)

        scores, index = torch.topk(scores.flatten(1), self.args.max_det, axis=-1)
        labels = index % nc
        index = torch.div(index, nc, rounding_mode='floor')
        bboxes = bboxes.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, bboxes.shape[-1]))
        
        preds = torch.cat([bboxes, scores.unsqueeze(-1), labels.unsqueeze(-1)], dim=-1)
        assert(preds.shape[0] == 1)
        mask = preds[..., 4] > self.args.conf
        preds = preds[mask].unsqueeze(0)

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i]
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            img_path = self.batch[0][i]
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))
        return results
