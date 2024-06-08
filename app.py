import gradio as gr
import cv2
import tempfile
from ultralytics import YOLOv10
import supervision as sv
from huggingface_hub import hf_hub_download


def download_models(model_id):
    hf_hub_download("kadirnar/Yolov10", filename=f"{model_id}", local_dir=f"./")
    return f"./{model_id}"
    
box_annotator = sv.BoxAnnotator()
category_dict = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
    6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
    11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
    16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
    22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag',
    27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard',
    32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
    36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
    40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
    46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
    51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake',
    56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table',
    61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard',
    67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink',
    72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors',
    77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
}


def yolov10_inference(image, video, model_id, image_size, conf_threshold, iou_threshold):
    model_path = download_models(model_id)
    model = YOLOv10(model_path)
    
    if image:
        results = model(source=image, imgsz=image_size, iou=iou_threshold, conf=conf_threshold, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        
        labels = [
            f"{category_dict[class_id]} {confidence:.2f}"
            for class_id, confidence in zip(detections.class_id, detections.confidence)
        ]
        annotated_image = box_annotator.annotate(image, detections=detections, labels=labels)
        return annotated_image[:, :, ::-1], None
    else:
        video_path = tempfile.mktemp(suffix=".webm")
        with open(video_path, "wb") as f:
            with open(video, "rb") as g:
                f.write(g.read())

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_video_path = tempfile.mktemp(suffix=".webm")
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'vp80'), fps, (frame_width, frame_height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(source=frame, imgsz=image_size, iou=iou_threshold, conf=conf_threshold, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)
            
            labels = [
                f"{category_dict[class_id]} {confidence:.2f}"
                for class_id, confidence in zip(detections.class_id, detections.confidence)
            ]
            annotated_frame = box_annotator.annotate(frame, detections=detections, labels=labels)
            out.write(annotated_frame)

        cap.release()
        out.release()

        return None, output_video_path


def yolov10_inference_for_examples(image, model_id, image_size, conf_threshold, iou_threshold):
    annotated_image, _ = yolov10_inference(image, None, model_id, image_size, conf_threshold, iou_threshold)
    return annotated_image


def app():
    with gr.Blocks():
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="pil", label="Image", visible=True)
                video = gr.Video(label="Video", visible=False)
                input_type = gr.Radio(
                    choices=["Image", "Video"],
                    value="Image",
                    label="Input Type",
                )
                model_id = gr.Dropdown(
                    label="Model",
                    choices=[
                        "yolov10n.pt",
                        "yolov10s.pt",
                        "yolov10m.pt",
                        "yolov10b.pt",
                        "yolov10l.pt",
                        "yolov10x.pt",
                    ],
                    value="yolov10m.pt",
                )
                image_size = gr.Slider(
                    label="Image Size",
                    minimum=320,
                    maximum=1280,
                    step=32,
                    value=640,
                )
                conf_threshold = gr.Slider(
                    label="Confidence Threshold",
                    minimum=0.1,
                    maximum=1.0,
                    step=0.1,
                    value=0.25,
                )
                iou_threshold = gr.Slider(
                    label="IoU Threshold",
                    minimum=0.1,
                    maximum=1.0,
                    step=0.1,
                    value=0.45,
                )
                yolov10_infer = gr.Button(value="Detect Objects")

            with gr.Column():
                output_image = gr.Image(type="numpy", label="Annotated Image", visible=True)
                output_video = gr.Video(label="Annotated Video", visible=False)

        def update_visibility(input_type):
            image_visibility = input_type == "Image"
            return (
                gr.update(visible=image_visibility),
                gr.update(visible=not image_visibility),
                gr.update(visible=image_visibility),
                gr.update(visible=not image_visibility),
            )

        input_type.change(
            fn=update_visibility,
            inputs=[input_type],
            outputs=[image, video, output_image, output_video],
        )

        def run_inference(image, video, model_id, image_size, conf_threshold, iou_threshold, input_type):
            if input_type == "Image":
                return yolov10_inference(image, None, model_id, image_size, conf_threshold, iou_threshold)
            else:
                return yolov10_inference(None, video, model_id, image_size, conf_threshold, iou_threshold)

        yolov10_infer.click(
            fn=run_inference,
            inputs=[image, video, model_id, image_size, conf_threshold, iou_threshold, input_type],
            outputs=[output_image, output_video],
        )

        gr.Examples(
            examples=[
                [
                    "ultralytics/assets/bus.jpg",
                    "yolov10s.pt",
                    640,
                    0.25,
                    0.45,
                ],
                [
                    "ultralytics/assets/zidane.jpg",
                    "yolov10s.pt",
                    640,
                    0.25,
                    0.45,
                ],
            ],
            fn=yolov10_inference_for_examples,
            inputs=[
                image,
                model_id,
                image_size,
                conf_threshold,
                iou_threshold,
            ],
            outputs=[output_image],
            cache_examples='lazy',
        )


gradio_app = gr.Blocks()
with gradio_app:
    gr.HTML(
        """
    <h1 style='text-align: center'>
    YOLOv10: Real-Time End-to-End Object Detection
    </h1>
    """)
    gr.HTML(
        """
        <h3 style='text-align: center'>
        <a href='https://arxiv.org/abs/2405.14458' target='_blank'>arXiv</a> | <a href='https://github.com/THU-MIG/yolov10' target='_blank'>github</a>
        </h3>
        """)
    with gr.Row():
        with gr.Column():
            app()

if __name__ == '__main__':
    gradio_app.launch()
