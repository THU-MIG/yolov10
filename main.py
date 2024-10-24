# import cv2
# from PIL import Image
# from ultralytics import YOLOv10
#
# # model = YOLOv10.from_pretrained('jameslahm/yolov10x')
# model = YOLOv10('model.onnx')
#
# # img1 = cv2.imread("ultralytics/assets/83.jpg")
# # img2 = cv2.imread("ultralytics/assets/101.jpg")
# # source1 = Image.open("ultralytics/assets/101.jpg")
# # source2 = Image.open("ultralytics/assets/83.jpg")
# img1 = "ultralytics/assets/83.jpg"
# img2 = "ultralytics/assets/101.jpg"
#
# results = model.predict([img1, img2], conf=0.35)
#
# for result in results:
#     result.show()
#
# print(results[0].tojson())


import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLOv10

# Ensure images are loaded and converted to NumPy arrays correctly
def load_image(image_path):
    img = Image.open(image_path)  # Load with PIL
    return np.array(img)  # Convert to NumPy array

# Load model
model = YOLOv10('model.onnx')

# Load images
img1 = load_image("ultralytics/assets/83.jpg")
img2 = load_image("ultralytics/assets/101.jpg")

# Predict
results = model.predict([img1, img2], conf=0.35)

# Show results
for result in results:
    result.show()

print(results[0].tojson())
