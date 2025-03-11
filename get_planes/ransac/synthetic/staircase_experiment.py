import sys
sys.path.append("/HighResMDE/get_planes/ransac")

import numpy as np
from information_estimation import plane_ransac
from visualise import visualise_mask, save_mask
from synthetic_test import set_depth, open3d_find_planes
from metrics import plane_ordering
from depth_to_pcd import depth_to_pcd

ROOT = "/HighResMDE/get_planes/staircase"
NOISE_LEVEL = 5

H = 480
W = 640
R = 10
EPSILON = 0.001
SIGMA = np.ones(H*W) * EPSILON * NOISE_LEVEL
MAX_PLANE = 8
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
gt = np.zeros((H,W),dtype=int)
N = 4
distance = [1.63,1,1,1.63]
distance = np.array(distance) * -(0.5)**0.5
for i in range(N):
    mask = np.zeros((H,W))
    mask[i*H//N:(i+1)*H//N,:] = 1
    gt[mask==1] = i+1

    angle = -(-1)**i * np.pi/4

    normal = np.array([0,np.sin(angle), np.cos(angle)])
    #normal = [0,0,1]

    depth += set_depth(np.ones((H,W)),INTRINSICS, mask, normal, distance[i])


#Add noise
depth += np.random.normal(0,5 * EPSILON,(H,W))

depth = np.array(depth/EPSILON,dtype=int) * EPSILON

print(depth.max(), depth.min())

import matplotlib.pyplot as plt
#plt.imsave(f"{ROOT}/staircase.png",depth,cmap='gray')
#visualise_mask(depth, np.zeros(H*W,dtype=int), INTRINSICS, filepath=f"{ROOT}/stair_gt.png",skip_color=True)
#visualise_mask(depth,gt, INTRINSICS, filepath=f"{ROOT}/stair_4.png")

mask, planes = open3d_find_planes(depth, INTRINSICS, EPSILON * NOISE_LEVEL, CONFIDENCE, INLIER_THRESHOLD, MAX_PLANE, verbose=True)
#save_mask(mask, f"{ROOT}/{NOISE_LEVEL}_default_stair.png")
#visualise_mask(depth, mask, INTRINSICS, filepath=f"{ROOT}/{NOISE_LEVEL}_default_pcd_stair.png")
visualise_mask(depth, mask, INTRINSICS)

R = depth.max() - depth.min()
print(R)
information, mask, planes = plane_ransac(depth, INTRINSICS, R, EPSILON, SIGMA, CONFIDENCE, INLIER_THRESHOLD, MAX_PLANE, verbose=True)
print(information)
print(planes)

smallest = np.argmin(information)
mask[mask>smallest] = 0
planes = planes[1:smallest+1]
print(mask.max())

points, index = depth_to_pcd(depth, INTRINSICS)
mask, planes = plane_ordering(points, mask, planes, R, EPSILON, SIGMA,keep_index=mask.max())

#save_mask(mask.reshape(H,W), f"{ROOT}/{NOISE_LEVEL}_ours_stair.png")
#visualise_mask(depth, mask, INTRINSICS, filepath=f"{ROOT}/{NOISE_LEVEL}_ours_pcd_stair.png")
visualise_mask(depth, mask, INTRINSICS)
