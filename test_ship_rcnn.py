from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2 import model_zoo
import os

from detectron2.evaluation import (
    COCOEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    verify_results,
)
from mask2former_video.data_video.ytvis_eval import YTVISEvaluator 

'''
This file tests registering the ship dataset to Detectron2's catalog

It then finetunes an out of the box Faster-RCNN model to the dataset to confirm that the dataset bounding boxes and segmentation is registered properly.
'''

class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to MaskFormer.
    """

    # Evaluator for our shipRS dataset does not exist
    # we are using COCO's evaluator, but that gives us 0.00 AP, issue??
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each
        builtin dataset. For your own dataset, you can simply create an
        evaluator manually in your script and do not have to worry about the
        hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        # instance segmentation
        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
            print(evaluator_list)
       
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)
      
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
  cfg.DATASETS.TRAIN = ("ship_train",)
  cfg.DATASETS.TEST = ("ship_val",)
  cfg.DATALOADER.NUM_WORKERS = 2
  cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl"  # Let training initialize from model zoo
  cfg.SOLVER.IMS_PER_BATCH = 2
  cfg.SOLVER.BASE_LR = 0.0025  # pick a good LR
  cfg.SOLVER.MAX_ITER = 5000    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
  cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster, and good enough for this toy dataset (default: 512)
  cfg.MODEL.ROI_HEADS.NUM_CLASSES = 50
  # cfg.TEST.EVAL_PERIOD = 50
  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
  cfg.OUTPUT_DIR = "./output/faster-rcnn"

  os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
  trainer = Trainer(cfg) 
  # trainer.build_evaluator(cfg, dataset_name='ship_val')
  trainer.resume_or_load(resume=False)
  trainer.train()