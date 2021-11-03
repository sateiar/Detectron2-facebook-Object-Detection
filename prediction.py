
import cv2
import os
import errno
import json

from detectron2.engine import DefaultPredictor
from detectron2.utils.logger import setup_logger
setup_logger()


def score_prediction(def_predictor: DefaultPredictor, image_url: str):
    print("===============Score prediction===============")
    im = cv2.imread(image_url)
    # make prediction
    return def_predictor(im)


def predict(img_folder, predictor: DefaultPredictor):
    print("===============Prediction Started===============")
    result = []
    lablelist = ['logo', 'table', 'stamp', 'signature']
    # predictor, classes = prepare_predictor()
    for file in os.listdir(img_folder):
        outputs = score_prediction(predictor, os.path.join(img_folder, file))
        temp = dict()
        array_1 = outputs["instances"].to("cpu").pred_boxes.tensor.numpy()
        array_2 = outputs["instances"].to("cpu").pred_classes.numpy()
        array_3 = outputs["instances"].to("cpu").scores.numpy()
        temp['file_name'] = file
        temp['objects'] = []
        for idx , item in enumerate(list(array_1)):
            my_dict = dict()
            my_dict['bbox'] = [int(item[0]), int(item[1]), int(item[2]), int(item[3])]
            my_dict['label'] = lablelist[int(list(array_2)[idx])]
            my_dict['prob'] = float(list(array_3)[idx])
            temp['objects'].append(my_dict)

        file_splited = file.split('.')[0].split('_')
        temp['page_no'] = file_splited[len(file_splited) - 1]

        result.append(temp)

    filename = os.path.join(img_folder, "result.json")
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    with open(filename, "w") as f:
        f.write(json.dumps(result))

    return result
