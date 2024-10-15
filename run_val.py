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

# Evaluate the model on the validation set
results = model.val(data='coco.yaml')

# Print the evaluation results
print(results)

# pred = results[0].boxes[0].conf

# # Hook to store the activations
# activations = {}

# def get_activation(name):
#     def hook(model, input, output):
#         activations[name] = output
#     return hook

# # Register hooks for each layer you want to inspect
# for name, layer in model.model.named_modules():
#     layer.register_forward_hook(get_activation(name))

# # Run the model to get activations
# results = model.predict(image_tensor, save=True, visualize=True)

# # # Print the activations
# # for name, activation in activations.items():
# #     print(f"Activation from layer {name}: {activation}")

# # List activation names separately
# print("\nActivation layer names:")
# for name in activations.keys():
#     print(name)
# # pred.backward()

# # Assuming 'model.23' is the layer of interest for bbox prediction and confidence
# activation = activations['model.23']['one2one'][0]
# act_23 = activations['model.23.cv3.2']
# act_dfl = activations['model.23.dfl.conv']
# act_conv = activations['model.0.conv']
# act_act = activations['model.0.act']

# # with torch.autograd.set_detect_anomaly(True):
# #     pred.backward()
# grad = torch.autograd.grad(act_23, im, grad_outputs=torch.ones_like(act_23), create_graph=True, retain_graph=True)[0]
# # grad = torch.autograd.grad(pred, im, grad_outputs=torch.ones_like(pred), create_graph=True)[0]
# grad = torch.autograd.grad(activations['model.23']['one2one'][1][0], 
#                            activations['model.23.one2one_cv3.2'], 
#                            grad_outputs=torch.ones_like(activations['model.23']['one2one'][1][0]), 
#                            create_graph=True, retain_graph=True)[0]

# # Print the results
# print(results)

# model.val(data='coco.yaml', batch=256)