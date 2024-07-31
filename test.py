import os
import time
import logging
import torch
import numpy as np
from gdino.model import OnnxGDINO
from  utils.gdino_utils import load_image, plot_boxes_to_image


logging.basicConfig(level=logging.INFO)

ogd = OnnxGDINO()

payload = ogd.preprocess_query("spaceman. spacecraft. water. clouds. space helmet")


output_dir = '/home/rhys'
img, img_transformed = load_image('/home/rhys/Downloads/wave_planet.webp')

img.save(os.path.join(output_dir, "pred.jpg"))



filtered_boxes, predicted_phrases = ogd.inference(img_transformed.astype(np.float32), 
                                                  payload,
                                                  text_threshold=0.25, 
                                                  box_threshold=0.35,)





size = img.size
pred_dict = {
    "boxes": filtered_boxes,
    "size": [size[1], size[0]],  # H,W
    "labels": predicted_phrases,
}

image_with_box = plot_boxes_to_image(img, pred_dict)[0]
image_with_box.save(os.path.join(output_dir, "pred.jpg"))
