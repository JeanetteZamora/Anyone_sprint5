
# Load here your Detection model
# The chosen detector model is "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
# because this particular model has a good balance between accuracy and speed.
# You can check the following Colab notebook with examples on how to run
# Detectron2 models
# https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5.
# Assign the loaded detection model to global variable DET_MODEL
# TODO

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from torchvision.ops import box_area

cfg = get_cfg()
#cfg.MODEL.DEVICE = "cpu" 
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
DET_MODEL = DefaultPredictor(cfg)

def get_vehicle_coordinates(img):
   """
    This function will run an object detector over the the image, get
    the vehicle position in the picture and return it.



    Parameters
    ----------
    img : numpy.ndarray
        Image in RGB format.

    Returns
    -------
    box_coordinates : tuple
        Tuple having bounding box coordinates as (left, top, right, bottom).
        Also known as (x1, y1, x2, y2).
   """
    # TODO
   image = DET_MODEL(img)
   # Many things should be taken into account to make it work:
   #      1. Current model being used can detect up to 80 different objects,
   #         we're only looking for 'cars' or 'trucks', so you should ignore
   #         other detected objects.

   mask = (image['instances'].pred_classes == 2) | (image['instances'].pred_classes == 7)

   #   2. The object detector may find more than one vehicle in the picture,
   #   you must then, choose the one with the largest area in the image.

   if image['instances'].pred_boxes[mask].tensor.size()[0] != 0:
      
      areas = box_area(image['instances'].pred_boxes[mask].tensor)
      
      max=0
      for j in range(areas.shape[0]):
         if areas[j]>max:
            max=areas[j]
            coord=image['instances'].pred_boxes[j]
            

         x1, y1, x2, y2 = coord.tensor.cpu().numpy()[0][:4]

         box_coordinates = (int(x1), int(y1), int(x2), int(y2))

   #   3. The model can also fail and detect zero objects in the picture,
   #   in that case, you should return coordinates that cover the full
   #   image, i.e. [0, 0, width, height].

   else:
      box_coordinates = [0,0,img.shape[1],img.shape[0]]

   # 4. Coordinates values must be integers, we're making reference to
   #    a position in a numpy.array, we can't use float values.

   return box_coordinates