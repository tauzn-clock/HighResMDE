import os
import json
import PIL
import matplotlib.pyplot as plt
import numpy as np
from depth_to_pcd import depth_to_pcd
from synthetic_test import set_depth, open3d_find_planes
from visualise import visualise_mask, save_mask
from information_estimation import plane_ransac
from metrics import plane_ordering

SOURCE_DIR = "/scratchdata/processed/alcove"
DEST_DIR = "/HighResMDE/get_planes/compare_img_2"

INDEX = 30

with open(f"{SOURCE_DIR}/camera_info.json", 'r') as f:
    config = json.load(f)

rgb = PIL.Image.open(f"{SOURCE_DIR}/rgb/{INDEX}.png")
depth = PIL.Image.open(f"{SOURCE_DIR}/depth/{INDEX+1}.png")

rgb = np.array(rgb)
depth = np.array(depth)/1000
print(depth.max(), depth.min())

plt.imsave(f"{DEST_DIR}/rgb.png", rgb)
plt.imsave(f"{DEST_DIR}/depth.png", depth, cmap='gray')

INTRINSICS = np.array(config["K"])
H, W = depth.shape
print(H, W)

EPSILON = 0.001
R = 10
CONFIDENCE = 0.99
INLIER_THRESHOLD = 0.1
MAX_PLANE = 32
NOISE_LEVEL = 20

points, index = depth_to_pcd(depth, INTRINSICS)

SIGMA = EPSILON*NOISE_LEVEL * points[:,2]
print(SIGMA.max(), SIGMA.min())

mask, open3d_planes = open3d_find_planes(depth, INTRINSICS, EPSILON*NOISE_LEVEL, CONFIDENCE, INLIER_THRESHOLD, MAX_PLANE, verbose=True)

save_mask(mask, f"{DEST_DIR}/{NOISE_LEVEL}_default_stair.png")
visualise_mask(depth, mask, INTRINSICS, filepath=f"{DEST_DIR}/{NOISE_LEVEL}_default_pcd_stair.png")

valid_mask = (depth > 0).flatten()
information, mask, planes = plane_ransac(depth, INTRINSICS, R, EPSILON, SIGMA, CONFIDENCE, INLIER_THRESHOLD, MAX_PLANE, valid_mask.flatten(), verbose=True)
print(information)
print(planes)

smallest = np.argmin(information)
mask[mask>smallest] = 0
planes = planes[1:smallest+1]
print(mask.max())

points, index = depth_to_pcd(depth, INTRINSICS)
mask, planes = plane_ordering(points, mask, planes, R, EPSILON, SIGMA,keep_index=mask.max())

save_mask(mask.reshape(H,W), f"{DEST_DIR}/{NOISE_LEVEL}_ours_stair.png")
visualise_mask(depth, mask, INTRINSICS, filepath=f"{DEST_DIR}/{NOISE_LEVEL}_ours_pcd_stair.png")
