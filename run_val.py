from ultralytics import YOLOv10
import torch
from PIL import Image
from torchvision import transforms

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model = YOLOv10.from_pretrained('jameslahm/yolov10{n/s/m/b/l/x}')
# model = YOLOv10.from_pretrained('jameslahm/yolov10n')
# or
# wget https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10{n/s/m/b/l/x}.pt
# wget https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10n.pt
# model = YOLOv10('yolov10{n/s/m/b/l/x}.pt')
model = YOLOv10('yolov10n.pt').to(device)

# Load the image
# path = '/home/nielseni6/PythonScripts/Github/yolov10/images/fat-dog.jpg'
path = '/home/nielseni6/PythonScripts/Github/yolov10/images/The-Cardinal-Bird.jpg'
image = Image.open(path)

# Define the transformation to resize the image, convert it to a tensor, and normalize it
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Apply the transformation
image_tensor = transform(image)

# Add a batch dimension
image_tensor = image_tensor.unsqueeze(0).to(device)
image_tensor = image_tensor.requires_grad_(True)


# Predict for a specific image
# results = model.predict(image_tensor, save=True)
# model.requires_grad_(True)


# for p in model.parameters():
#     p.requires_grad = True
results = model.predict(image_tensor, save=True)

# Display the results
for result in results:
    print(result)

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