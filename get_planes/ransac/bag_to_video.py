import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from tqdm import tqdm

def depth_to_color(depth):
    depth = depth / depth.max()
    H, W = depth.shape
    color = np.zeros((H, W, 3))
    for i in range(H):
        for j in range(W):
            color[i,j] = plt.cm.viridis(depth[i,j])[0:3]
    color[depth == 0] = 0
    return color

DIR_FILE = "/scratchdata/alcove"
N = 406 // 4
CONVERT_DEPTH = True
CREATE_COMBINED = True

if CONVERT_DEPTH:
    DEPTH_COLOR = os.path.join(DIR_FILE, "depth_color")
    if not os.path.exists(DEPTH_COLOR):
        os.makedirs(DEPTH_COLOR)
        print(f"Directory '{DEPTH_COLOR}' created.")
    for i in range(N):
        depth = Image.open(os.path.join(DIR_FILE, "depth", f"{i}.png"))
        depth = np.array(depth)
        depth = depth_to_color(depth)
        plt.imsave(f"{DIR_FILE}/depth_color/{i}.png", depth)

if CREATE_COMBINED:
    COMBINED = os.path.join(DIR_FILE, "combined")
    if not os.path.exists(COMBINED):
        os.makedirs(COMBINED)
        print(f"Directory '{COMBINED}' created.")
    for i in range(N):
        img = plt.imread(os.path.join(DIR_FILE, "rgb", f"{i}.png"))
        depth = plt.imread(os.path.join(DIR_FILE, "depth_color", f"{i}.png"))
        open3d = plt.imread(os.path.join(DIR_FILE, "open3d", f"{i}.png"))
        our = plt.imread(os.path.join(DIR_FILE, "our_color", f"{i}.png"))

        H, W, _ = img.shape

        combined = np.zeros((2*H, 2*W, 3))

        combined[:H, :W] = img
        combined[:H, W:] = depth[:,:,:3]
        combined[H:, :W] = open3d[:,:,:3]
        combined[H:, W:] = our[:,:,:3]

        plt.imsave(f"{DIR_FILE}/combined/{i}.png", combined)