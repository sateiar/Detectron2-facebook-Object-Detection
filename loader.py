import os
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor


def prepare_predictor(model_name='model_final.pth'):
    # create config
    cfg = get_cfg()
    # below path applies to current installation location of Detectron2
    cfg.merge_from_file(model_zoo.get_config_file(os.path.join("COCO-Detection", "faster_rcnn_R_101_FPN_3x.yaml")))
    cfg.MODEL.WEIGHTS = os.path.join(os.path.dirname(__file__), model_name)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
    cfg.MODEL.DEVICE = "cpu"

    classes = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes
    predictor = DefaultPredictor(cfg)
    print("Predictor has been initialized with classes.", classes)
    return predictor