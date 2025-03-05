import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

def depth_to_color(depth):
    depth = depth / depth.max()
    H, W = depth.shape
    color = np.zeros((H, W, 3))
    for i in range(H):
        for j in range(W):
            color[i,j] = plt.cm.viridis(depth[i,j])[0:3]
    color[depth == 0] = 0
    return color

DIR_FILE = "/scratchdata/corridor"
N = 406

all_imgs = []

for i in range(N):
    img = plt.imread(os.path.join(DIR_FILE, "rgb", f"{i}.png"))
    depth = Image.open(os.path.join(DIR_FILE, "depth", f"{i}.png"))
    depth = np.array(depth)
    open3d = plt.imread(os.path.join(DIR_FILE, "open3d", f"{i}.png"))
    our = plt.imread(os.path.join(DIR_FILE, "our_color", f"{i}.png"))

    H, W, _ = img.shape

    combined = np.zeros((2*H, 2*W, 3))

    combined[:H, :W] = img
    combined[:H, W:] = depth_to_color(depth)
    combined[H:, :W] = open3d[:,:,:3]
    combined[H:, W:] = our[:,:,:3]

    all_imgs.append(combined)

    plt.imsave(f"{DIR_FILE}/combined/{i}.png", combined)

frame_rate = 12
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for MP4 format
out = cv2.VideoWriter(f'{DIR_FILE}/combined.mp4', fourcc, frame_rate, (2*W, 2*H))

for img in all_imgs:
    img = (img*255).astype(np.uint8)
    #Flip bgr to rgb
    img = img[:,:,::-1]
    out.write(img)

out.release()