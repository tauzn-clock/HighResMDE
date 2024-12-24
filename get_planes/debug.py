import os
import csv
import numpy as np
from PIL import Image
import open3d as o3d

DIR_PATH = "/scratchdata/nyu_depth_v2/sync"
FILE_PATH = "/HighResMDE/src/nddepth_train.csv"

data = []

with open(FILE_PATH, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        for i in range(2, 7):
            row[i] = float(row[i])

        data.append(row)

i = 14320

intrinsic = np.array([[data[i][2]/1000, 0, data[i][4], 0],
                      [0, data[i][3]/1000, data[i][5], 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
intrinsic = intrinsic.flatten()
print(intrinsic)


depth = Image.open(os.path.join(DIR_PATH, data[i][1]))
depth = np.array(depth) / data[i][6]
print(depth.max())


def depth_to_pcd(depth_image, intrinsic, ):
    # Get dimensions of the depth image
    height, width = depth_image.shape

    # Generate a grid of (x, y) coordinates
    x, y = np.meshgrid(np.arange(width), np.arange(height))

    # Flatten the arrays
    x = x.flatten()
    y = y.flatten()
    depth = depth_image.flatten()

    # Calculate 3D coordinates
    fx, fy, cx, cy = intrinsic[0], intrinsic[5], intrinsic[2], intrinsic[6]
    z = depth

    x_3d = (x - cx) * z / fx
    y_3d = (y - cy) * z / fy

    # Create a point cloud
    points = np.vstack((x_3d, y_3d, z)).T
    return points

cood_3d = depth_to_pcd(depth, intrinsic)

print(cood_3d[:4000, :].shape)

# Create an Open3D point cloud object
point_cloud = o3d.geometry.PointCloud()

# Assign the NumPy array to the point cloud
point_cloud.points = o3d.utility.Vector3dVector(cood_3d)
#o3d.visualization.draw_geometries([pcd])
