import numpy as np
from information_estimation import plane_ransac
from visualise import visualise_mask, save_mask
from synthetic_test import set_depth, open3d_find_planes
import matplotlib.pyplot as plt
import open3d as o3d
def pcd_to_img(pcd):
    # Visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Add the point cloud to the visualizer
    vis.add_geometry(pcd)

    opt = vis.get_render_option()
    opt.point_size = 2

    view_control = vis.get_view_control()
    view_control.set_zoom(0.6) 
    view_control.rotate(100.0, 100.0)

    return np.array(vis.capture_screen_float_buffer(True))

NOISE_LEVEL = 5

ROOT = "/HighResMDE/get_planes/4_planes"
H = 480
W = 640
R = 10
EPSILON = 0.001
SIGMA = np.ones(H*W) * EPSILON * NOISE_LEVEL
MAX_PLANE = 10
CONFIDENCE = 0.99
INLIER_THRESHOLD = 0.20

INTRINSICS = np.array([500, 0, 320, 0, 0, 500, 240])
N = 4
depth = np.zeros((H,W))
mask = np.zeros((H,W),dtype=int)
for i in range(N):
    plane_mask = np.zeros((H,W),dtype=bool)
    plane_mask[:,i*W//N:(i+1)*W//N] = True
    depth += set_depth(np.ones((H,W)),INTRINSICS, plane_mask, [0,0,1], -0.1 * i - 2.5)
    mask[plane_mask] = i+1

# Random assignment
random_mask = np.random.randint(100, size=(H,W))
cutoff = 40
depth[random_mask>cutoff] = np.random.randint(0.24*R//EPSILON, 0.29*R//EPSILON, size=(H,W))[random_mask>cutoff] * EPSILON
mask[random_mask>cutoff] = 0

print(depth.max(), depth.min())

import matplotlib.pyplot as plt
plt.imsave(f"{ROOT}/our.png",depth)
visualise_mask(depth, np.zeros(H*W,dtype=int), INTRINSICS, filepath=f"{ROOT}/our_gt.png",skip_color=True)

mask, planes = open3d_find_planes(depth, INTRINSICS, EPSILON * NOISE_LEVEL, CONFIDENCE, INLIER_THRESHOLD, MAX_PLANE, verbose=True)
save_mask(mask, f"{ROOT}/{NOISE_LEVEL}_default_our.png")
visualise_mask(depth, mask, INTRINSICS, filepath=f"{ROOT}/{NOISE_LEVEL}_default_pcd_our.png")

R = depth.max() - depth.min()
print(R)
information, mask, planes = plane_ransac(depth, INTRINSICS, R, EPSILON, SIGMA, CONFIDENCE, INLIER_THRESHOLD, MAX_PLANE, verbose=True)
print(information)
print(planes)

smallest = np.argmin(information)

mask[mask>smallest] = 0
print(mask.max())

save_mask(mask.reshape(H,W), f"{ROOT}/{NOISE_LEVEL}_ours_our.png")
visualise_mask(depth, mask, INTRINSICS, filepath=f"{ROOT}/{NOISE_LEVEL}_ours_pcd_our.png")

