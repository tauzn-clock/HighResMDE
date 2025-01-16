from process_depth_img import depth_to_pcd
from information_estimation import default_ransac
import open3d as o3d
import numpy as np
import csv
import os
from PIL import Image

root = "/scratchdata/nyu_depth_v2/sync"
data_csv = "/HighResMDE/src/nddepth_train_v2.csv"

with open(data_csv, 'r') as f:
    reader = csv.reader(f)
    data = list(reader)

data = data[0]

EPSILON = 1/float(data[6]) # Resolution
R = float(data[7]) # Maximum Range
SIGMA = EPSILON  # Normal std

CONFIDENCE = 0.999
INLIER_THRESHOLD = 0.01
MAX_PLANE = 4

INTRINSICS = [float(data[2]), 0, float(data[4]), 0, 0, float(data[3]), float(data[5]), 0] # fx, fy, cx, cy
H = 480
W = 640

depth = Image.open(os.path.join(root, data[1]))
depth = np.array(depth)

valid_mask = depth > 0

points, index = depth_to_pcd(depth, INTRINSICS)

information, mask, plane = default_ransac(points, R, EPSILON, SIGMA, CONFIDENCE, INLIER_THRESHOLD, MAX_PLANE, valid_mask.flatten())

for i in range(MAX_PLANE+1):
    print(f"Cnt {i}", np.sum(mask[i]==i))
print("Planes: ", plane)

print("Information:", information)

# Visualize the point cloud
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)
color = np.zeros((points.shape[0], 3))
#Find index of smallest information
min_idx = np.argmin(information)
print("Found Planes", min_idx)

for i in range(1, min_idx+1):
    color[mask[i]==i] = np.random.rand(3)
point_cloud.colors = o3d.utility.Vector3dVector(color)

o3d.visualization.draw_geometries([point_cloud])

