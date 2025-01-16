from test_pcd import get_plane
from information_estimation import default_ransac
import open3d as o3d
import numpy as np

EPSILON = 0.1 # Resolution
R = 10 # Maximum Range
SIGMA = EPSILON * 4 # Normal std

CONFIDENCE = 0.995
INLIER_THRESHOLD = 0.5
MAX_PLANE = 1

point_cloud = get_plane(R, EPSILON)

points = np.asarray(point_cloud.points)
print(points.shape)

mask, plane = default_ransac(points, R, EPSILON, SIGMA, CONFIDENCE, INLIER_THRESHOLD, MAX_PLANE)

point_cloud.points = o3d.utility.Vector3dVector(points[mask])

# Visualize the point cloud
o3d.visualization.draw_geometries([point_cloud])
