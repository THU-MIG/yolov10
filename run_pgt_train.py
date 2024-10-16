from ultralytics import YOLOv10, YOLO
# from ultralytics.engine.pgt_trainer import PGTTrainer
import os
from ultralytics.models.yolo.segment import PGTSegmentationTrainer
import argparse


def main(args):
  # model = YOLOv10()

  # If you want to finetune the model with pretrained weights, you could load the 
  # pretrained weights like below
  # model = YOLOv10.from_pretrained('jameslahm/yolov10{n/s/m/b/l/x}')
  # or
  # wget https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10{n/s/m/b/l/x}.pt
  model = YOLOv10('yolov10n.pt', task='segment')

  args = dict(model='yolov10n.pt', data='coco.yaml', 
              epochs=args.epochs, batch=args.batch_size,
              # cfg = 'pgt_train.yaml', # This can be edited for full control of the training process
              )
  trainer = PGTSegmentationTrainer(overrides=args)
  trainer.train(
        # debug=True, 
        #   args = dict(pgt_coeff=0.1), # Should add later to config
          )

  # Save the trained model
  model.save('yolov10_coco_trained.pt')

  # Evaluate the model on the validation set
  results = model.val(data='coco.yaml')

  # Print the evaluation results
  print(results)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Train YOLOv10 model with PGT segmentation.')
  parser.add_argument('--device', type=str, default='0', help='CUDA device number')
  parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
  parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for training')
  args = parser.parse_args()

  # Set CUDA device (only needed for multi-gpu machines)
  os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
  os.environ["CUDA_VISIBLE_DEVICES"] = args.device
  main(args)