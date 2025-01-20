from process_depth_img import depth_to_pcd
from information_estimation import default_ransac, plane_ransac
import open3d as o3d
import numpy as np
import csv
import os
from PIL import Image
import matplotlib.pyplot as plt

root = "/scratchdata/nyu_depth_v2/sync"#"/scratchdata/processed/stair_up"
data_csv = "/HighResMDE/src/nddepth_train_v2.csv"

with open(data_csv, 'r') as f:
    reader = csv.reader(f)
    data = list(reader)

data = data[104]
#data = ["rgb/1.png", "depth/1.png", 306.75604248046875, 306.7660827636719, 322.9314270019531, 203.91506958007812, 1, 2**16]

INTRINSICS = [float(data[2]), 0, float(data[4]), 0, 0, float(data[3]), float(data[5]), 0] # fx, fy, cx, cy
INTRINSICS = np.array(INTRINSICS)

depth = Image.open(os.path.join(root, data[1]))
depth = np.array(depth) /float(data[6])
H, W = depth.shape

valid_mask = depth > 0

EPSILON = 1/float(data[6]) # Resolution
print("EPSILON", EPSILON)
R = float(data[7]) # Maximum Range
print("R", R)
SIGMA = EPSILON * 5 # Normal std

CONFIDENCE = 0.999
INLIER_THRESHOLD = 5e4/(H*W)
MAX_PLANE = 6

points, index = depth_to_pcd(depth, INTRINSICS)
#information, mask, plane = default_ransac(points, R, EPSILON, SIGMA, CONFIDENCE, INLIER_THRESHOLD, MAX_PLANE, valid_mask.flatten())
information, mask, plane = plane_ransac(depth, INTRINSICS, R, EPSILON, SIGMA, CONFIDENCE, INLIER_THRESHOLD, MAX_PLANE, valid_mask.flatten())

for i in range(MAX_PLANE+1):
    print(f"Cnt {i}", np.sum(mask[i]==i))
print("Planes: ", plane)

#Find index of smallest information
min_idx = np.argmin(information)
print("Found Planes", min_idx)

print("Information:", information)

dist = points @ plane[1:min_idx+1,:3].T + np.stack([plane[1:min_idx+1,3]]*points.shape[0], axis=0)
dist = np.abs(dist)
isPartofPlane = mask != 0
mask = np.argmin(dist, axis=1) + 1
mask = mask * isPartofPlane

# Visualize the point cloud
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)
color = np.zeros((points.shape[0], 3))

for i in range(1, min_idx+1):
    color[mask[i]==i] = np.random.rand(3)
point_cloud.colors = o3d.utility.Vector3dVector(color)

o3d.visualization.draw_geometries([point_cloud])
