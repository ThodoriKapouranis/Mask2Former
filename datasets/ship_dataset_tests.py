from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2 import model_zoo
import os 

'''
This file tests registering the ship dataset to Detectron2's catalog

It then finetunes an out of the box Faster-RCNN model to the dataset to confirm that the dataset bounding boxes and segmentation is registered properly.
'''

if __name__ == "__main__":
  dataset = 'datasets/ShipRSImageNet'
  
  # Test to see if dataset load properly
  register_coco_instances(
    "ship_train", 
    {}, 
    f'{dataset}/annotations/train.json',
    f'{dataset}images'
  )
  
  register_coco_instances(
    "ship_test", 
    {}, 
    f'{dataset}/annotations/test.json',
    f'{dataset}images'
  )
  
  register_coco_instances(
    "ship_val", 
    {}, 
    f'{dataset}/annotations/val.json',
    f'{dataset}images'
  )
  
  dataset = DatasetCatalog.get('ship_train')

  print('Datasets created succesfully')
  
  # Attempt to use an out of the box faster-RCNN to finetune to the model
  cfg = get_cfg()
  cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")) #Get the basic model configuration from the model zoo 
  #Passing the Train and Validation sets
  cfg.DATASETS.TRAIN = ("my_dataset_train",)
  cfg.DATASETS.TEST = ()
  # Number of data loading threads
  cfg.DATALOADER.NUM_WORKERS = 4
  cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
  # Number of images per batch across all machines.
  cfg.SOLVER.IMS_PER_BATCH = 4
  cfg.SOLVER.BASE_LR = 0.02  # pick a good LearningRate
  cfg.SOLVER.MAX_ITER = 5000  #No. of iterations   
  cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512 
  cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2 # No. of classes = [text] ONLY
  cfg.TEST.EVAL_PERIOD = 100 # No. of iterations after which the Validation Set is evaluated. 
  os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
  trainer = DefaultTrainer(cfg) 
  trainer.resume_or_load(resume=False)