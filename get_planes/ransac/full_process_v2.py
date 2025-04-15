import sys
sys.path.append('/HighResMDE/segment-anything')

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from depth_to_pcd import depth_to_pcd
from information_estimation import plane_ransac
import numpy as np
import csv
import os
from PIL import Image
import time
import json
import matplotlib.pyplot as plt
from visualise import visualise_pcd, visualise_mask

#Set seed
np.random.seed(0)

DEVICE="cuda:0"
root = "/scratchdata/nyu_plane"
with open(os.path.join(root, "camera_info.json"), "r") as f:
    camera_info = json.load(f)
INTRINSICS = camera_info["P"]
print(INTRINSICS)

TARGET_FOLDER = "sam"

EPSILON = 0.001 # Resolution
R = 10.0 # Maximum Range

SIGMA_RATIO = 0.01

CONFIDENCE = 0.99
INLIER_THRESHOLD = 0.1
MAX_PLANE = 16

USE_SAM = True

SAM_CONFIDENCE = 0.99
SAM_INLIER_THRESHOLD = 0.2
SAM_MAX_PLANE = 4

POST_PROCESSING = False

if USE_SAM: 
    sam = sam_model_registry["default"](checkpoint="/scratchdata/sam_vit_h_4b8939.pth").to(DEVICE)
    mask_generator = SamAutomaticMaskGenerator(sam, stability_score_thresh=0.98)

for frame_cnt in range(0,1000):
    img = Image.open(os.path.join(root, f"rgb/{frame_cnt}.png"))
    img = np.array(img)

    depth = Image.open(os.path.join(root, f"depth/{frame_cnt}.png"))
    depth = np.array(depth) * EPSILON
    H, W = depth.shape

    points, index = depth_to_pcd(depth, INTRINSICS)
    SIGMA = SIGMA_RATIO * points[:,2]
    print(SIGMA.max(), SIGMA.min())
    #SIGMA = 0.0012 + 0.0019 * (points[:,2] - 0.4)**2
    #print(SIGMA.max(), SIGMA.min())

    global_mask = np.zeros((H, W), dtype=int).flatten()
    global_planes = []

    if USE_SAM:
        sam_masks = mask_generator.generate(img)
        print(len(sam_masks))
        masks = sorted(sam_masks, key=lambda x: x["stability_score"])

        

        for sam_i, sam_mask in enumerate(sam_masks):
            valid_mask = sam_mask["segmentation"] & (depth > 0)

            information, mask, plane = plane_ransac(depth, INTRINSICS, R, EPSILON, SIGMA, SAM_CONFIDENCE, SAM_INLIER_THRESHOLD, SAM_MAX_PLANE, valid_mask.flatten(),verbose=False,post_processing=POST_PROCESSING)

            min_idx = np.argmin(information)
            print("Min Planes: ", min_idx)
            for i in range(1, min_idx+1):
                global_mask[mask==i] = global_mask.max() + 1
                global_planes.append(plane[i])

    # Remaining points
    valid_mask = (global_mask == 0) & (depth > 0).flatten()
    information, mask, plane = plane_ransac(depth, INTRINSICS, R, EPSILON, SIGMA, CONFIDENCE, INLIER_THRESHOLD, MAX_PLANE, valid_mask.flatten(), verbose=True, post_processing=POST_PROCESSING)

    min_idx = np.argmin(information)
    print("Min Planes: ", min_idx)
    for i in range(1, min_idx+1):
        global_mask[mask==i] = global_mask.max() + 1
        global_planes.append(plane[i])


    #Save the planes
    # Save the mask
    global_mask = global_mask.reshape(H, W).astype(np.uint8)
    print(global_mask.max())
    plt.imsave("mask.png", global_mask)

    mask_PIL = Image.fromarray(global_mask)
    mask_PIL.save(os.path.join(root, TARGET_FOLDER, f"{frame_cnt}.png"))

    # Save the plane
    with open(os.path.join(root, TARGET_FOLDER, f"{frame_cnt}.csv"), 'w') as f:
        writer = csv.writer(f)
        writer.writerows(global_planes)

visualise_mask(depth, global_mask, INTRINSICS)