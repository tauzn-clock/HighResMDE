import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

DIR_FILE = "/scratchdata/processed/corridor"
N = 24

all_imgs = []

for i in range(N):
    img = plt.imread(os.path.join(DIR_FILE, "rgb", f"{i}.png"))
    depth = plt.imread(os.path.join(DIR_FILE, "depth", f"{i}.png"))
    open3d = plt.imread(os.path.join(DIR_FILE, "open3d", f"{i}.png"))
    our = plt.imread(os.path.join(DIR_FILE, "our", f"{i}.png"))

    print(img.shape,depth.shape,open3d.shape,our.shape)

    H, W, _ = img.shape

    combined = np.zeros((2*H, 2*W, 3))

    combined[:H, :W] = img
    combined[:H, W:,0] = depth
    combined[:H, W:,1] = depth
    combined[:H, W:,2] = depth
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