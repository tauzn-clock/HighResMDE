from process_depth_img import depth_to_pcd
import numpy as np
import open3d as o3d
from PIL import Image
import csv
import os

def visualise_mask(depth, mask, intrinsics):
    points, _ = depth_to_pcd(depth, intrinsics) 

    # Visualize the point cloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    color = np.zeros((points.shape[0], 3))

    mask = mask.flatten()
    for i in range(1, mask.max()+1):
        color[mask==i] = np.random.rand(3)
    point_cloud.colors = o3d.utility.Vector3dVector(color)

    o3d.visualization.draw_geometries([point_cloud])

if __name__ == '__main__':
    root = "/scratchdata/nyu_plane"
    data_csv = "/HighResMDE/get_planes/ransac/config/nyu.csv"

    with open(data_csv, 'r') as f:
        reader = csv.reader(f)
        DATA = list(reader)

    INDEX = 0

    data = DATA[INDEX]

    INTRINSICS = [float(data[2]), 0, float(data[4]), 0, 0, float(data[3]), float(data[5]), 0] # fx, fy, cx, cy
    INTRINSICS = np.array(INTRINSICS)

    depth = np.array(Image.open(os.path.join(root, data[1])))/float(data[6])
    mask = np.array(Image.open(os.path.join(root, "original_gt", f"{INDEX}.png")))
    mask = np.array(Image.open(os.path.join(root, "new_gt", f"{INDEX}.png")))

    visualise_mask(depth, mask, INTRINSICS)