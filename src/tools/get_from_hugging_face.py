from datasets import load_dataset
import numpy as np
from PIL import Image
import torch

ds = load_dataset("sayakpaul/nyu_depth_v2")

print(ds.keys())

#data["image"] is in RGB, just save as is
#data["depth_map"] is in [0,10] meters, map to [0, 2**16-1], save as uint16

max_depth = 0
min_depth = 100

store_path = []

for i, data in enumerate(ds["validation"]):

    img = data["image"]
    img.save(f"/scratchdata/nyu_huggingface/test/rgb_{i}.png")

    depth = data["depth_map"]
    depth = torch.Tensor(np.array(depth))
    depth = depth * (2**16 - 1) / 10
    depth = torch.round(depth)
    #depth = depth.type(torch.uint16)
    depth = depth.numpy().astype(np.uint16)
    depth = Image.fromarray(depth)
    depth.save(f"/scratchdata/nyu_huggingface/test/depth_{i}.png")

    store_path.append([f"test/rgb_{i}.png", f"test/depth_{i}.png"])

import csv

with open("/scratchdata/nyu_huggingface/test.csv", mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(store_path)

test = Image.open("depth_0.png")
test = np.array(test)
print(test.max(), test.min())