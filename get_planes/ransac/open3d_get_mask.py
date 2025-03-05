import os
import numpy as np
import open3d as o3d
from synthetic_test import open3d_find_planes
import json
from PIL import Image
import cv2
from depth_to_pcd import depth_to_pcd
from visualise import visualise_mask, save_mask
import matplotlib.pyplot as plt
from information_estimation import plane_ransac
from metrics import plane_ordering

FILE_DIR = "/scratchdata/processed/corridor"

with open(os.path.join(FILE_DIR, "camera_info.json"), "r") as f:
    camera_info = json.load(f)

INTRINSICS = camera_info["K"]
print(INTRINSICS)

NOISE_LEVEL = 10
R = 10
EPSILON = 0.001
CONFIDENCE = 0.99
INLIER_THRESHOLD = 0.1
MAX_PLANE = 16

for index in range(380,406):

    depth = Image.open(os.path.join(FILE_DIR, "depth", f"{index}.png"))
    depth = np.array(depth) / 1000

    plt.imsave("test.png", depth, cmap='gray')

    H, W = depth.shape

    #mask, planes = open3d_find_planes(depth, INTRINSICS, EPSILON * NOISE_LEVEL, CONFIDENCE, INLIER_THRESHOLD, MAX_PLANE, verbose=True)

    #save_mask(mask.reshape(H,W), f"{FILE_DIR}/open3d/{index}.png")

    SIGMA = points, index = depth_to_pcd(depth, INTRINSICS)
    SIGMA = 0.01 * points[:,2]


    valid_mask = (depth > 0).flatten()
    information, mask, planes = plane_ransac(depth, INTRINSICS, R, EPSILON, SIGMA, CONFIDENCE, INLIER_THRESHOLD, MAX_PLANE, valid_mask.flatten(), verbose=True)
    print(information)
    print(planes)

    smallest = np.argmin(information)
    mask[mask>smallest] = 0
    planes = planes[1:smallest+1]
    print(mask.max())

    points, index = depth_to_pcd(depth, INTRINSICS)
    mask, planes = plane_ordering(points, mask, planes, R, EPSILON, SIGMA)
    print(mask.max())   
    print(planes)

    save_mask(mask.reshape(H,W), f"{FILE_DIR}/our/{index}.png")