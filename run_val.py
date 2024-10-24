from ultralytics import YOLOv10, YOLO, YOLOv10PGT
# from ultralytics.engine.pgt_trainer import PGTTrainer
import os
from ultralytics.models.yolo.segment import PGTSegmentationTrainer
import argparse
from datetime import datetime

# nohup python run_pgt_train.py --device 1 > ./output_logs/gpu1_yolov10_pgt_train.log 2>&1 & 

def main(args):

    model = YOLOv10PGT(args.model_path)
    
    # Evaluate the model on the validation set
    results = model.val(data=args.data_yaml)
    
    # Print the evaluation results
    print(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train YOLOv10 model with PGT segmentation.')
    parser.add_argument('--device', type=str, default='1', help='CUDA device number')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for training')
    parser.add_argument('--data_yaml', type=str, default='coco.yaml', help='Path to the data YAML file')
    parser.add_argument('--model_path', type=str, default='yolov10n.pt', help='Path to the model file')
    args = parser.parse_args()

    # Set CUDA device (only needed for multi-gpu machines)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    main(args)