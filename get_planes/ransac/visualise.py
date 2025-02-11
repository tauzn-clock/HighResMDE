from depth_to_pcd import depth_to_pcd
import numpy as np
import open3d as o3d
from PIL import Image
import csv
import os
from metrics import plane_ordering

def visualise_mask(depth, mask, intrinsics, index=None):
    points, _ = depth_to_pcd(depth, intrinsics) 
    visualise_pcd(points, mask, index)

def visualise_pcd(points, mask, index=None):
    def hsv_to_rgb(h, s, v):
        """
        Convert HSV to RGB.

        :param h: Hue (0 to 360)
        :param s: Saturation (0 to 1)
        :param v: Value (0 to 1)
        :return: A tuple (r, g, b) representing the RGB color.
        """
        h = h / 360  # Normalize hue to [0, 1]
        c = v * s  # Chroma
        x = c * (1 - abs((h * 6) % 2 - 1))  # Temporary value
        m = v - c  # Match value

        if 0 <= h < 1/6:
            r, g, b = c, x, 0
        elif 1/6 <= h < 2/6:
            r, g, b = x, c, 0
        elif 2/6 <= h < 3/6:
            r, g, b = 0, c, x
        elif 3/6 <= h < 4/6:
            r, g, b = 0, x, c
        elif 4/6 <= h < 5/6:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x

        # Adjust to match value
        r = (r + m) 
        g = (g + m) 
        b = (b + m)

        return r, g, b
    
    if index is not None: INDEX = index
    else: INDEX = mask.max()

    # Visualize the point cloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    color = np.zeros((points.shape[0], 3))

    mask = mask.flatten()
    for i in range(1, INDEX+1):
        color[mask==i] = hsv_to_rgb(i/INDEX*360, 1, 1)
    point_cloud.colors = o3d.utility.Vector3dVector(color)

    o3d.visualization.draw_geometries([point_cloud])

if __name__ == '__main__':
    root = "/scratchdata/nyu_plane"
    data_csv = "/HighResMDE/get_planes/ransac/config/nyu.csv"
    SIGMA_RATIO = 0.01
    FOLDER = "new_gt_20240205"

    with open(data_csv, 'r') as f:
        reader = csv.reader(f)
        DATA = list(reader)

    INDEX = 0

    data = DATA[INDEX]

    EPSILON = 1/float(data[6]) # Resolution
    R = float(data[7]) # Maximum Range

    INTRINSICS = np.array([float(data[2]), 0, float(data[4]), 0, 0, float(data[3]), float(data[5]), 0]) # fx, fy, cx, cy

    depth = np.array(Image.open(os.path.join(root, data[1])))/float(data[6])
    mask = np.array(Image.open(os.path.join(root, "original_gt", f"{INDEX}.png")))
    mask = np.array(Image.open(os.path.join(root, FOLDER, f"{INDEX}.png")))

    points, index = depth_to_pcd(depth, INTRINSICS)
    SIGMA = SIGMA_RATIO * points[:,2]

    with open(os.path.join(root, FOLDER, f"{INDEX}.csv"), 'r') as f:
        reader = csv.reader(f)
        csv_data = np.array(list(reader), dtype=np.float32)

    mask = mask.flatten()
    mask, csv_data = plane_ordering(points, mask, csv_data, R, EPSILON, SIGMA,keep_index=8)

    visualise_mask(depth, mask, INTRINSICS)