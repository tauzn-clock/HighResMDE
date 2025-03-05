import os
from visualise import visualise_mask, save_mask
from PIL import Image
import numpy as np

DIR_PATH = "/scratchdata/corridor"

for index in range(400):

    mask = Image.open(f"{DIR_PATH}/our/{index}.png")
    mask = np.array(mask)
    H,W = mask.shape
    print(mask.max(),mask.min())
    save_mask(mask.reshape(H,W), f"{DIR_PATH}/our_color/{index}.png")