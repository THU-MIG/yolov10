from ultralytics import YOLOv10, YOLO
# from ultralytics.engine.pgt_trainer import PGTTrainer
# from ultralytics import BaseTrainer
# from ultralytics.engine.trainer import BaseTrainer
import os
from ultralytics.models.yolo.segment import PGTSegmentationTrainer


# Set CUDA device (only needed for multi-gpu machines) 
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "4" 

# model = YOLOv10()
# model = YOLO('yolov8n-seg.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

# model = YOLO()
# If you want to finetune the model with pretrained weights, you could load the 
# pretrained weights like below
# model = YOLOv10.from_pretrained('jameslahm/yolov10{n/s/m/b/l/x}')
# or
# wget https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10{n/s/m/b/l/x}.pt
model = YOLOv10('yolov10n.pt', task='segment')

args = dict(model='yolov10n.pt', data='coco128-seg.yaml')
trainer = PGTSegmentationTrainer(overrides=args)
trainer.train(
            # debug=True, 
            #   args = dict(pgt_coeff=0.1),
              )

# model.train(
#             # data='coco.yaml', 
#             data='coco128-seg.yaml', 
#             trainer=model._smart_load("pgt_trainer"), # This is needed to generate attributions (will be used later to train via PGT)
#             # Add return_images as input parameter
#             epochs=500, batch=16, imgsz=640,
#             debug=True, # If debug = True, the attributions will be saved in the figures folder
#             # cfg='/home/nielseni6/PythonScripts/yolov10/ultralytics/cfg/models/v8/yolov8-seg.yaml',
#             # overrides=dict(task="segment"),
#             )

# Save the trained model
model.save('yolov10_coco_trained.pt')

# Evaluate the model on the validation set
results = model.val(data='coco.yaml')

# Print the evaluation results
print(results)