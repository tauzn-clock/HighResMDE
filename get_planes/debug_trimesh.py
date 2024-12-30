import os
import json
import csv
import numpy as np
from PIL import Image
import open3d as o3d
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import cv2

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

def check_directories_exist(file_path):
    # Split the file path into directories
    directories = os.path.normpath(file_path).split(os.sep)
    
    # Iterate through all directory levels
    for i in range(2, len(directories)):
        current_dir = os.sep.join(directories[:i])  # Get the path for the current level
        
        # Check if the directory exists
        if os.path.isdir(current_dir):
            print(f"Directory exists: {current_dir}")
        else:
            print(f"Directory does not exist: {current_dir}")
            os.mkdir(current_dir)

DIR_PATH = "/scratchdata/nyu_depth_v2/sync"
FILE_PATH = "/HighResMDE/src/nddepth_train.csv"

data = []

with open(FILE_PATH, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        for i in range(2, 7):
            row[i] = float(row[i])

        data.append(row)

for i in range(len(data)):

    intrinsic = np.array([[data[i][2], 0, data[i][4], 0],
                        [0, data[i][3], data[i][5], 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
    intrinsic = intrinsic.flatten()


    depth = Image.open(os.path.join(DIR_PATH, data[i][1]))
    depth = np.array(depth) / data[i][6]

    coord_3d, index = depth_to_pcd(depth, intrinsic)
    index = index[coord_3d[:,2] > 0]
    coord_3d = coord_3d[coord_3d[:,2] > 0]

    # Create an Open3D point cloud object
    pcd = o3d.geometry.PointCloud()

    o3d.visualization.draw_geometries([pcd])
    break