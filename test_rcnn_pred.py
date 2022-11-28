from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.utils.visualizer import ColorMode
from detectron2.data import DatasetCatalog, MetadataCatalog
import cv2
from detectron2 import model_zoo
import os
from detectron2.data.datasets import register_coco_instances

if __name__ == "__main__":
  dataset = 'datasets/ShipRSImageNet'
  
  # Test to see if dataset load properly
  register_coco_instances(
    "ship_train", 
    {}, 
    f'{dataset}/annotations/train.json',
    f'{dataset}/images'
  )
  
  register_coco_instances(
    "ship_val", 
    {}, 
    f'{dataset}/annotations/val.json',
    f'{dataset}/images'
  )
    
  register_coco_instances(
    "ship_test", 
    {}, 
    f'{dataset}/annotations/test.json',
    f'{dataset}/images'
  )
  
  dataset = DatasetCatalog.get('ship_train')
  print('Datasets created succesfully')
  
  # Attempt to use an out of the box faster-RCNN to finetune to the model
  cfg = get_cfg()
  cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
  cfg.OUTPUT_DIR = "./output/faster-rcnn"
  cfg.DATASETS.TRAIN = ("ship_train",)
  cfg.DATASETS.TEST = ("ship_val",)
  cfg.DATALOADER.NUM_WORKERS = 2
  cfg.SOLVER.IMS_PER_BATCH = 2
  cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
  cfg.SOLVER.MAX_ITER = 1000    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
  cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster, and good enough for this toy dataset (default: 512)
  cfg.MODEL.ROI_HEADS.NUM_CLASSES = 50
  # cfg.TEST.EVAL_PERIOD = 50
  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1
  cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
  
  predictor = DefaultPredictor(cfg)
  
  val = DatasetCatalog.get('ship_val')
  
  data = val[0]
  im = cv2.imread(data['file_name'])
  outputs = predictor(im)
  
  print(f'\n Predictions: {outputs}')
  print(f'\n Original data: {data}')
  
  
  
  
