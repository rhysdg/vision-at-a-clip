
import os
import time
import logging
import torch
import numpy as np
from gdino.model import OnnxGDINO
from  utils.gdino_utils import load_image, viz

logging.basicConfig(level=logging.INFO)

output_dir = 'output'

ogd = OnnxGDINO(type='gdino_fp32', 
                trt=True, 
                warmup=True)

payload = ogd.preprocess_query("spaceman. spacecraft. water. clouds. space helmet. glove")
img, img_transformed = load_image('images/wave_planet.webp')

img.save(os.path.join(output_dir, "pred.jpg"))


start = time.time()
filtered_boxes, predicted_phrases = ogd.inference(img_transformed.astype(np.float32), 
                                                  payload,
                                                  text_threshold=0.25, 
                                                  box_threshold=0.35,)

end = time.time()

start = time.time()
filtered_boxes, predicted_phrases = ogd.inference(img_transformed.astype(np.float32), 
                                                  payload,
                                                  text_threshold=0.25, 
                                                  box_threshold=0.35,)

end = time.time()

print((end-start)*1000)


size = img.size
pred_dict = {
    "boxes": filtered_boxes,
    "size": [size[1], size[0]],
    "labels": predicted_phrases,
}

predictions = viz(img, 
                  pred_dict,
                  label_size=25,
                  bbox_thickness=6
                  )[0]

predictions.save(os.path.join(output_dir, "pred.jpg"))

