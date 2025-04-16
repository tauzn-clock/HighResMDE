import cv2
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from information_estimation import plane_ransac
from depth_to_pcd import depth_to_pcd
from PIL import Image

DIR_PATH = "/home//scratchdata/processed/outdoor"
TARGET_FOLDER = "canny"
with open(os.path.join(DIR_PATH, "camera_info.json"), "r") as f:
    camera_info = json.load(f)
INTRINSICS = camera_info["P"]
print(INTRINSICS)

EPSILON = 0.001 # Resolution
R = 10.0 # Maximum Range

SIGMA_RATIO = 0.01

CONFIDENCE = 0.99
INLIER_THRESHOLD = 0.1
MAX_PLANE = 16

POST_PROCESSING = False

for INDEX in range(0,1000):
    img = cv2.imread(os.path.join(DIR_PATH, "rgb", f"{INDEX}.png"))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    depth = cv2.imread(os.path.join(DIR_PATH, "depth", f"{INDEX}.png"), cv2.IMREAD_UNCHANGED)
    depth = depth.astype(np.float32) * EPSILON  # Convert to meters
    H, W = depth.shape

    # Canny edge detection
    canny_edges = cv2.Canny(gray, 100, 400)
    canny_edges = cv2.morphologyEx(canny_edges, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
    print(canny_edges.shape, canny_edges.max(), canny_edges.min())

    edges_inv = cv2.bitwise_not(canny_edges)

    num_labels, labels = cv2.connectedComponents(edges_inv, connectivity=4)
    print(num_labels, labels.shape, labels.max(), labels.min())

    # Pixels with labels 0 are filled with label of closest pixel
    segment_mask = labels.copy()
    zero_indices = np.argwhere(segment_mask == 0)
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1),
                (-1, -1), (-1, 1), (1, -1), (1, 1)]
    while len(zero_indices)!=0:
        new_zero_indices = []
        new_segment_mask = segment_mask.copy()
        for i, j in zero_indices:
            # Get the 8-connected neighbors
            filled = False
            for ni, nj in neighbors:
                index_i = i + ni
                index_j = j + nj
                if 0 <= index_i < segment_mask.shape[0] and 0 <= index_j < segment_mask.shape[1]:
                    if segment_mask[index_i, index_j] > 0:
                        new_segment_mask[i, j] = segment_mask[index_i, index_j]
                        filled = True
                        break
            if not filled:
                new_zero_indices.append((i, j))
        zero_indices = np.array(new_zero_indices) 
        segment_mask = new_segment_mask.copy()
        print(len(zero_indices))

    global_mask = np.zeros((H, W), dtype=int).flatten()
    global_planes = []

    points, index = depth_to_pcd(depth, INTRINSICS)
    SIGMA = SIGMA_RATIO * points[:,2]

    for i in range(1, segment_mask.max()+1):
        if (segment_mask == i).sum() < H*W*0.05:
            continue
        valid_mask = (segment_mask==i) & (depth > 0)

        information, mask, plane = plane_ransac(depth, INTRINSICS, R, EPSILON, SIGMA, CONFIDENCE, INLIER_THRESHOLD, MAX_PLANE, valid_mask.flatten(),verbose=False,post_processing=POST_PROCESSING)

        min_idx = np.argmin(information)
        print("Min Planes: ", min_idx)
        for i in range(1, min_idx+1):
            global_mask[mask==i] = global_mask.max() + 1
            global_planes.append(plane[i])

    # Save the mask
    global_mask = global_mask.reshape(H, W).astype(np.uint8)
    print(global_mask.max())
    plt.imsave("mask.png", global_mask)

    mask_PIL = Image.fromarray(global_mask)
    mask_PIL.save(os.path.join(DIR_PATH, TARGET_FOLDER, f"{INDEX}.png"))