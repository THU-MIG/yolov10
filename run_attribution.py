from ultralytics import YOLOv10, YOLO
# from ultralytics.engine.pgt_trainer import PGTTrainer
# from ultralytics import BaseTrainer
# from ultralytics.engine.trainer import BaseTrainer
import os

# Set CUDA device (only needed for multi-gpu machines) 
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "4" 

# model = YOLOv10()
# model = YOLO()
# If you want to finetune the model with pretrained weights, you could load the 
# pretrained weights like below
# model = YOLOv10.from_pretrained('jameslahm/yolov10{n/s/m/b/l/x}')
# or
# wget https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10{n/s/m/b/l/x}.pt
model = YOLOv10('yolov10n.pt')

model.train(data='coco.yaml', 
            trainer=model._smart_load("pgt_trainer"), # This is needed to generate attributions (will be used later to train via PGT)
            # Add return_images as input parameter
            epochs=500, batch=16, imgsz=640,
            debug=True, # If debug = True, the attributions will be saved in the figures folder
            )

# Save the trained model
model.save('yolov10_coco_trained.pt')

# Evaluate the model on the validation set
results = model.val(data='coco.yaml')

# Print the evaluation results
print(results)