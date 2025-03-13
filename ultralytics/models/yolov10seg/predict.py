from ultralytics.models.yolo.detect import DetectionPredictor
import torch
from ultralytics.utils import ops
from ultralytics.engine.results import Results
import torch.nn.functional as F

class YOLOv10SegPredictor(DetectionPredictor):
    def postprocess(self, preds, img, orig_imgs):
        coef,proto = None,None
        if isinstance(preds, dict):
            coef = preds['coef']
            proto = preds['proto']
            preds = preds["one2one"]
        if isinstance(preds, (list, tuple)):
            preds = preds[0]  # [1,5,6006]  coef[1,32,6006]  proto[1,32,104,176]

        if preds.shape[-1] == 6:
            pass
        else:
            preds = preds.transpose(-1, -2)  # [1,6006,5]
            coef = coef.transpose(-1, -2)
            # bboxes, scores, labels = ops.v10postprocess(preds, self.args.max_det, preds.shape[-1]-4)
            bboxes, scores, labels,segmask = ops.v10segpostprocess([preds,coef,proto], self.args.max_det, preds.shape[-1]-4)
            bboxes = ops.xywh2xyxy(bboxes)
            preds = torch.cat([bboxes, scores.unsqueeze(-1), labels.unsqueeze(-1)], dim=-1)
        mask = preds[..., 4] > self.args.conf
        if self.args.classes is not None:
            mask = mask & (preds[..., 5:6] == torch.tensor(self.args.classes, device=preds.device).unsqueeze(0)).any(2)
        
        preds = [p[mask[idx]] for idx, p in enumerate(preds)]
        segmask = [p[mask[idx]] for idx, p in enumerate(segmask)]
        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i]
            seg = segmask[i]
            cc,hh,ww = seg.shape
            seg = F.interpolate(seg[None], (hh*4, ww*4), mode="bilinear", align_corners=False)[0].gt_(0)            
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            
            img_path = self.batch[0][i]
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred,masks=seg))
        return results
