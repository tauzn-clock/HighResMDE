from test_pcd import get_plane
from information_estimation import default_ransac
import open3d as o3d
import numpy as np

EPSILON = 0.1 # Resolution
R = 10 # Maximum Range
SIGMA = 0.0001 # Normal std

CONFIDENCE = 0.995
INLIER_THRESHOLD = 0.25
MAX_PLANE = 1

point_cloud = get_plane(R, EPSILON)

points = np.asarray(point_cloud.points)
print(points.shape)

default_ransac(points, R, EPSILON, SIGMA, CONFIDENCE, INLIER_THRESHOLD, MAX_PLANE)

# Visualize the point cloud
o3d.visualization.draw_geometries([point_cloud])
