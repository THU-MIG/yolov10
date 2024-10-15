from ultralytics import YOLOv10, YOLO
# from ultralytics.engine.pgt_trainer import PGTTrainer
# from ultralytics import BaseTrainer
# from ultralytics.engine.trainer import BaseTrainer
import os

# Set CUDA device (only needed for multi-gpu machines) 
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "4" 

model = YOLOv10()
# model = YOLO()
# If you want to finetune the model with pretrained weights, you could load the 
# pretrained weights like below
# model = YOLOv10.from_pretrained('jameslahm/yolov10{n/s/m/b/l/x}')
# or
# wget https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10{n/s/m/b/l/x}.pt
# model = YOLOv10('yolov10m.pt')

model.train(data='coco.yaml', 
            # Add return_images as input parameter
            epochs=500, batch=16, imgsz=640,
            )

# Save the trained model
model.save('yolov10_coco_trained.pt')

# Evaluate the model on the validation set
results = model.val(data='coco.yaml')

# Print the evaluation results
print(results)

# import torch
# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms

# # Define the transformation for the dataset
# transform = transforms.Compose([
#     transforms.Resize((640, 640)),
#     transforms.ToTensor()
# ])

# # Load the COCO dataset
# train_dataset = datasets.CocoDetection(root='data/nielseni6/coco/train2017', annFile='/data/nielseni6/coco/annotations/instances_train2017.json', transform=transform)
# val_dataset = datasets.CocoDetection(root='data/nielseni6/coco/val2017', annFile='/data/nielseni6/coco/annotations/instances_val2017.json', transform=transform)

# # Create data loaders
# train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)
# val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=4)

# model = YOLOv10()

# # Define the optimizer
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# # Training loop
# for epoch in range(500):
#     model.train()
#     for images, targets in train_loader:
#         images = images.to('cuda')
#         targets = [{k: v.to('cuda') for k, v in t.items()} for t in targets]
#         loss = model(images, targets)
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()

#     # Validation loop
#     model.eval()
#     with torch.no_grad():
#         for images, targets in val_loader:
#             images = images.to('cuda')
#             targets = [{k: v.to('cuda') for k, v in t.items()} for t in targets]
#             results = model(images, targets)

# # Save the trained model
# model.save('yolov10_coco_trained.pt')

# # Evaluate the model on the validation set
# results = model.val(data='coco.yaml')

# # Print the evaluation results
# print(results)