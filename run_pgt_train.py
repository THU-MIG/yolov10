from ultralytics import YOLOv10, YOLO, YOLOv10PGT
# from ultralytics.engine.pgt_trainer import PGTTrainer
import os
from ultralytics.models.yolo.segment import PGTSegmentationTrainer
import argparse
from datetime import datetime
import torch

# nohup python run_pgt_train.py --device 0 > ./output_logs/gpu0_yolov10_pgt_train.log 2>&1 & 

def main(args):
    model = YOLOv10PGT('yolov10n.pt')
    model.train(    
                data=args.data_yaml, 
                epochs=args.epochs, 
                batch=args.batch_size,
                # amp=False,
                # pgt_coeff=1.5,
                # cfg='pgt_train.yaml',  # Load and train model with the config file
                )
    # If you want to finetune the model with pretrained weights, you could load the 
    # pretrained weights like below 
    # model = YOLOv10.from_pretrained('jameslahm/yolov10{n/s/m/b/l/x}')
    # or
    # wget https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10{n/s/m/b/l/x}.pt
    # model = YOLOv10('yolov10n.pt', task='segment')
    # model = YOLOv10('yolov10n.pt', task='segment')

    # args_dict = dict(
    #             model='yolov10n.pt', 
    #             data=args.data_yaml, 
    #             epochs=args.epochs, batch=args.batch_size,
    #             # pgt_coeff=5.0,
    #             # cfg = 'pgt_train.yaml', # This can be edited for full control of the training process
    #             )
    # trainer = PGTSegmentationTrainer(overrides=args_dict)
    # trainer.train(
    #             # debug=True, 
    #             # args = dict(pgt_coeff=0.1), # Should add later to config
    #             )

    # Create a directory to save model weights if it doesn't exist
    model_weights_dir = 'model_weights'
    if not os.path.exists(model_weights_dir):
        os.makedirs(model_weights_dir)

    # Save the trained model with a unique name based on the current date and time
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    data_yaml_base = os.path.splitext(os.path.basename(args.data_yaml))[0]
    model_save_path = os.path.join(model_weights_dir, f'yolov10_{data_yaml_base}_trained_{current_time}.pt')
    model.save(model_save_path)
    # torch.save(trainer.model.state_dict(), model_save_path)
    

    # Evaluate the model on the validation set
    results = model.val(data=args.data_yaml)

    # Print the evaluation results
    print(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train YOLOv10 model with PGT segmentation.')
    parser.add_argument('--device', type=str, default='0', help='CUDA device number')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for training')
    parser.add_argument('--data_yaml', type=str, default='coco.yaml', help='Path to the data YAML file')
    args = parser.parse_args()

    # Set CUDA device (only needed for multi-gpu machines)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    main(args)
    