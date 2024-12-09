from PIL import Image
import numpy as np
from torchvision import transforms
import torch
import os
import re
import csv

IMG_PATH_TRAIN = "/scratchdata/nyu_depth_v2/official_splits/train/basement/sync_depth_00489.png"
IMG_PATH_TEST = "/scratchdata/nyu_depth_v2/official_splits/test/bathroom/sync_depth_00045.png"

train_img = Image.open(IMG_PATH_TRAIN)
test_img = Image.open(IMG_PATH_TEST)

train_img = torch.Tensor(np.array(train_img)).unsqueeze(0)
test_img = torch.Tensor(np.array(test_img)).unsqueeze(0)

print(train_img.max(), train_img.min())
print(test_img.max(), test_img.min())


DIR_TRAIN = "/scratchdata/nyu_depth_v2/sync"
#DIR_TRAIN = "/scratchdata/nyu_depth_v2/official_splits/train"
DIR_TEST = "/scratchdata/nyu_depth_v2/official_splits/test"

# Specify the directory path

# Use a list to keep track of directories to be processed
directories_to_process = [DIR_TEST]
store_path = []

while directories_to_process:
    # Pop a directory from the list to process
    current_dir = directories_to_process.pop()

    # Iterate through all items in the current directory
    for item in os.listdir(current_dir):
        item_path = os.path.join(current_dir, item)

        # Check if it's a directory
        if os.path.isdir(item_path):
            print(item_path)
            #print(f"{item_path} is a directory")
            # Add directory to the list to process later
            directories_to_process.append(item_path)
        else:
            #print(f"{item_path} is a file")
            if re.search("/rgb_", item_path):
                store_path.append(item_path)

store_csv_path = "/HighResMDE/src/nyu_test_v2.csv"
store_csv = []
focal = 518.8579
fx = focal
fy = focal
cx = 325.5824
cy = 253.7362
depth_scale = 1000
depth_max = 10

for img_path in store_path:
    print(img_path)
    depth_path = re.sub("/rgb_", "/sync_depth_", img_path, flags=re.IGNORECASE)
    depth_path = re.sub(".jpg", ".png", depth_path, flags=re.IGNORECASE)
    # Check if the depth path exists
    if not os.path.exists(depth_path):
        print(f"Depth path {depth_path} does not exist")
        continue

    store_csv.append([img_path, depth_path, fx, fy, cx, cy, depth_scale, depth_max])

print(len(store_csv))

with open(store_csv_path, mode='w') as file:
    writer = csv.writer(file)
    writer.writerows(store_csv)