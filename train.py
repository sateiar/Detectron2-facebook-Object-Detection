# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import os
import random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import load_coco_json

# register COCO dataset if it is in COCO format:
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog
from datetime import datetime
now = datetime.now()
strdate=now.strftime("%m%d%Y%H%M%S")
coco_model = "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
output_path = 'output'
number_class = 4
def training(json_path,images_path,strdate,coco_model,output_path,number_class):
    coco_name= "train_"+strdate

    register_coco_instances(coco_name, {}, json_path,
                            images_path)

    # metadata = MetadataCatalog.get(coco_name)
    # dataset_dicts = DatasetCatalog.get(coco_name)

    # create a detectron2 config and a detectron2 DefaultTrainer to run training
    cfg = get_cfg()

    cfg.merge_from_file(model_zoo.get_config_file(coco_model))
    cfg.DATASETS.TRAIN = (coco_name,)
    cfg.DATASETS.TEST = ()
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(coco_model) # initialize training from model zoo

    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.WARMUP_ITERS = 1000
    cfg.SOLVER.MAX_ITER = 1500
    cfg.SOLVER.STEPS = (1000, 1500)
    cfg.SOLVER.GAMMA = 0.05
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = number_class
    cfg.TEST.EVAL_PERIOD = 500
    cfg.OUTPUT_DIR = output_path

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    return





