from process_depth_img import depth_to_pcd
from information_estimation import default_ransac, plane_ransac
import open3d as o3d
import numpy as np
import csv
import os
from PIL import Image
import time
import matplotlib.pyplot as plt
from post_processing import post_processing
from test_pcd import get_plane
import json

#Set seed
np.random.seed(0)

root = "/scratchdata/nyu_plane"
data_csv = "/HighResMDE/get_planes/ransac/config/nyu.csv"

with open(data_csv, 'r') as f:
    reader = csv.reader(f)
    DATA = list(reader)

with open(os.path.join(root,"new_gt","param.json"), 'r') as f:
    PARAM = json.load(f)

for frame_cnt in range(10,len(DATA)):
    data = DATA[frame_cnt]
    #data = ["rgb/90.png", "depth/90.png", 306.75604248046875, 306.7660827636719, 322.9314270019531, 203.91506958007812, 1, 2**16]

    INTRINSICS = [float(data[2]), 0, float(data[4]), 0, 0, float(data[3]), float(data[5]), 0] # fx, fy, cx, cy
    INTRINSICS = np.array(INTRINSICS)

    depth = Image.open(os.path.join(root, data[1]))
    depth = np.array(depth) /float(data[6])
    H, W = depth.shape
    #depth = get_plane(H,W,INTRINSICS)

    START = time.time()

    valid_mask = depth > 0

    EPSILON = 1/float(data[6]) # Resolution
    R = float(data[7]) # Maximum Range
    #SIGMA = np.ones(H*W) * EPSILON * 20 # Normal std

    CONFIDENCE = PARAM["confidence"]
    INLIER_THRESHOLD = PARAM["inlier_cnt"]/(H*W)
    MAX_PLANE = PARAM["max_plane"]

    points, index = depth_to_pcd(depth, INTRINSICS)
    SIGMA = PARAM["sigma_ratio"] * points[:,2]
    #SIGMA = 2 * points[:,2]**2 + 1.4 * points[:,2] + 1.1057
    #SIGMA *= 1e-3
    #print(SIGMA.max(), SIGMA.min())
    #information, mask, plane = default_ransac(points, R, EPSILON, SIGMA, CONFIDENCE, INLIER_THRESHOLD, MAX_PLANE, valid_mask.flatten())
    information, mask, plane = plane_ransac(depth, INTRINSICS, R, EPSILON, SIGMA, CONFIDENCE, INLIER_THRESHOLD, MAX_PLANE, valid_mask.flatten(),verbose=False)

    print("Time Taken: ", time.time()-START)

    print("Total Points: ", valid_mask.sum())

    for i in range(1,MAX_PLANE+1):
        print(f"Cnt {i}", np.sum(mask==i))
    print("Planes: ", plane)

    #Find index of smallest information
    min_idx = np.argmin(information)
    print("Found Planes", min_idx)

    print("Information:", information)

    # Post Processing
    information, mask, plane = post_processing(depth, INTRINSICS, R, EPSILON, SIGMA, information, mask, plane, valid_mask)

    #Find index of smallest information
    min_idx = np.argmin(information)
    print("Found Planes", min_idx)

    print("Information:", information)

    # Save the mask
    print(mask.dtype)
    mask = mask.reshape(H, W).astype(np.uint8)

    plt.imsave("mask.png", mask)

    mask_PIL = Image.fromarray(mask)
    mask_PIL.save(os.path.join(root, "new_gt", f"{frame_cnt}.png"))

    # Save the plane
    with open(os.path.join(root, "new_gt", f"{frame_cnt}.csv"), 'w') as f:
        writer = csv.writer(f)
        writer.writerows(plane)


