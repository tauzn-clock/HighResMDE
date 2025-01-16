from test_pcd import get_plane
from information_estimation import default_ransac
import open3d as o3d
import numpy as np

EPSILON = 0.1 # Resolution
R = 10 # Maximum Range
SIGMA = EPSILON * 0.5 # Normal std

CONFIDENCE = 0.999
INLIER_THRESHOLD = 0.5
MAX_PLANE = 1

points = get_plane(R, EPSILON)

print(points.shape)

information, mask, plane = default_ransac(points, R, EPSILON, SIGMA, CONFIDENCE, INLIER_THRESHOLD, MAX_PLANE)

print("Information:", information)
for i in range(MAX_PLANE+1):
    print(f"Cnt {i}", np.sum(mask[i]==i))
print("Planes: ", plane)

# Visualize the point cloud
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)
color = np.zeros((points.shape[0], 3))
for i in range(1,MAX_PLANE+1):
    color[mask[i]==i] = np.random.rand(3)
point_cloud.colors = o3d.utility.Vector3dVector(color)

o3d.visualization.draw_geometries([point_cloud])
