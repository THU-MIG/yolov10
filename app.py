# Ackownledgement: https://huggingface.co/spaces/kadirnar/Yolov10/blob/main/app.py
# Thanks to @kadirnar

import gradio as gr
from ultralytics import YOLOv10 

def yolov10_inference(image, model_path, image_size, conf_threshold):
    model = YOLOv10(model_path)
    
    model.predict(source=image, imgsz=image_size, conf=conf_threshold, save=True)
    
    return model.predictor.plotted_img[:, :, ::-1]

def app():
    with gr.Blocks():
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="pil", label="Image")
                
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
                    value="yolov10s.pt",
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
                    minimum=0.0,
                    maximum=1.0,
                    step=0.1,
                    value=0.25,
                )
                yolov10_infer = gr.Button(value="Detect Objects")

            with gr.Column():
                output_image = gr.Image(type="numpy", label="Annotated Image")

        yolov10_infer.click(
            fn=yolov10_inference,
            inputs=[
                image,
                model_id,
                image_size,
                conf_threshold,
            ],
            outputs=[output_image],
        )

        gr.Examples(
            examples=[
                [
                    "ultralytics/assets/bus.jpg",
                    "yolov10s.pt",
                    640,
                    0.25,
                ],
                [
                    "ultralytics/assets/zidane.jpg",
                    "yolov10s.pt",
                    640,
                    0.25,
                ],
            ],
            fn=yolov10_inference,
            inputs=[
                image,
                model_id,
                image_size,
                conf_threshold,
            ],
            outputs=[output_image],
            cache_examples=True,
        )

gradio_app = gr.Blocks()
with gradio_app:
    gr.HTML(
        """
    <h1 style='text-align: center'>
    YOLOv10: Real-Time End-to-End Object Detection
    </h1>
    """)
    with gr.Row():
        with gr.Column():
            app()

gradio_app.launch(debug=True)