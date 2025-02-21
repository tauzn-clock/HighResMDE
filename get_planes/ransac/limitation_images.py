import numpy as np
from information_estimation import plane_ransac
from visualise import visualise_mask, save_mask
from synthetic_test import set_depth, open3d_find_planes
import matplotlib.pyplot as plt
import open3d as o3d
from metrics import plane_ordering
from depth_to_pcd import depth_to_pcd

NOISE_LEVEL = 10

ROOT = "/HighResMDE/get_planes/limitation"

H = 480
W = 640
R = 10
EPSILON = 0.001
SIGMA = np.ones(H*W) * EPSILON * NOISE_LEVEL
MAX_PLANE = 2
CONFIDENCE = 0.99
INLIER_THRESHOLD = 0.25
INTRINSICS = np.array([500, 0, W//2, 0, 0, 500, H//2])

Z = np.ones((H,W)).flatten()

x, y = np.meshgrid(np.arange(W), np.arange(H))
x = x.flatten()
y = y.flatten()
fx, fy, cx, cy = INTRINSICS[0], INTRINSICS[5], INTRINSICS[2], INTRINSICS[6]
x_3d = (x - cx) * Z / fx
y_3d = (y - cy) * Z / fy
POINTS = np.vstack((x_3d, y_3d, Z)).T
DIRECTION_VECTOR = POINTS / (np.linalg.norm(POINTS, axis=1)[:, None]+1e-7)

DIRECTION_VECTOR = DIRECTION_VECTOR.reshape(H,W,3)

depth = np.zeros((H,W))
gt = np.zeros((H,W))
N = 2
normal = [[0.707,0, 0.707],[-0.707, 0, 0.707]]
distance = [-1,-2]

mask = np.zeros((H,W))
mask[:,:int(0.5*W)] = 1
gt[mask==1] = 0+1
depth += set_depth(np.ones((H,W)),INTRINSICS, mask, normal[0], distance[0])

mask = np.zeros((H,W))
mask[:,int(0.5*W):] = 1
gt[mask==1] = 1+1
depth += set_depth(np.ones((H,W)),INTRINSICS, mask, normal[1], distance[1])

depth = np.array(depth/EPSILON,dtype=int) * EPSILON

#visualise_mask(depth, np.zeros_like(depth,dtype=np.int8), INTRINSICS)

plt.imsave(f"{ROOT}/limitation.png",depth,cmap='gray')
visualise_mask(depth, np.zeros(H*W,dtype=int), INTRINSICS, filepath=f"{ROOT}/stair_gt.png",skip_color=True)

R = depth.max() - depth.min()
print(R)
information, mask, planes = plane_ransac(depth, INTRINSICS, R, EPSILON, SIGMA, CONFIDENCE, INLIER_THRESHOLD, MAX_PLANE, verbose=True, post_processing=True)#, post_processing=True)
print(information)
print(planes)

smallest = np.argmin(information)
mask[mask>smallest] = 0
planes = planes[1:smallest+1]
print(mask.max())

#points, index = depth_to_pcd(depth, INTRINSICS)
#mask, planes = plane_ordering(points, mask, planes, R, EPSILON, SIGMA,keep_index=mask.max())

save_mask(mask.reshape(H,W), f"{ROOT}/{NOISE_LEVEL}_ours_stair.png")
visualise_mask(depth, mask, INTRINSICS, filepath=f"{ROOT}/{NOISE_LEVEL}_ours_pcd_stair.png")
