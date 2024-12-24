import os
import csv
import numpy as np
from PIL import Image
import open3d as o3d
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage

DIR_PATH = "/scratchdata/nyu_depth_v2/sync"
FILE_PATH = "/HighResMDE/src/nddepth_train.csv"

DISTANCE_THESHOLD=0.02
NUM_ITERATION=10000
PROB=0.9995

data = []

with open(FILE_PATH, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        for i in range(2, 7):
            row[i] = float(row[i])

        data.append(row)

i = 14320

intrinsic = np.array([[data[i][2], 0, data[i][4], 0],
                      [0, data[i][3], data[i][5], 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
intrinsic = intrinsic.flatten()


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
    
    index = np.column_stack((x, y))

    return points, index

coord_3d, index = depth_to_pcd(depth, intrinsic)
index = index[coord_3d[:,2] > 0]
coord_3d = coord_3d[coord_3d[:,2] > 0]

# Create an Open3D point cloud object
pcd = o3d.geometry.PointCloud()

# Assign the NumPy array to the point cloud
pcd.points = o3d.utility.Vector3dVector(coord_3d)
#o3d.visualization.draw_geometries([pcd])

mask = np.zeros(depth.shape)
plane_params = []

for k in range(7):
    plane_model, inliers = pcd.segment_plane(distance_threshold=DISTANCE_THESHOLD,
                                                ransac_n=3,
                                                num_iterations=NUM_ITERATION,
                                                probability=PROB)
    print(plane_model)

    plane_params.append(plane_model)

    tmp_mask = np.zeros(mask.shape)
    store_inliers = np.zeros(mask.shape).astype(int)
    
    for i in inliers:
        tmp_mask[index[i][1], index[i][0]] = 1
        store_inliers[index[i][1], index[i][0]] = i

    labeled_mask, num_features = ndimage.label(tmp_mask)

    region_sizes = ndimage.sum(tmp_mask, labeled_mask, range(1, num_features + 1))
    largest_region_label = np.argmax(region_sizes) + 1 
    largest_region_mask = labeled_mask == largest_region_label
    plt.imsave("largest.png", largest_region_mask)

    deleted_pts = store_inliers[largest_region_mask]

    for i in deleted_pts:
        mask[index[i][1], index[i][0]] = k+1
    
    new_coord_3d = np.delete(coord_3d, deleted_pts, axis=0)
    new_index = np.delete(index, deleted_pts, axis=0)

    coord_3d = new_coord_3d
    index = new_index

    pcd.points = o3d.utility.Vector3dVector(coord_3d)
    #o3d.visualization.draw_geometries([pcd])

plt.imsave("mask.png", mask)


