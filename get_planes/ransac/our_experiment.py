import numpy as np
from information_estimation import plane_ransac
from visualise import visualise_mask
from synthetic_test import set_depth, open3d_find_planes
H = 480
W = 640
R = 10
EPSILON = 0.001
SIGMA = np.ones(H*W) * 0.002
MAX_PLANE = 8
CONFIDENCE = 0.99
INLIER_THRESHOLD = 0.15

INTRINSICS = np.array([500, 0, 320, 0, 0, 500, 240])
N = 4
depth = np.zeros((H,W))
for i in range(N):
    mask = np.zeros((H,W),dtype=bool)
    mask[:,i*W//N:(i+1)*W//N] = True
    depth += set_depth(np.ones((H,W)),INTRINSICS, mask, [0,0,1], -0.1 * i - 2.5)
depth += np.random.randint(10, size=(H,W)) * EPSILON

# Random assignment
random_mask = np.random.randint(2, size=(H,W))
depth[random_mask==0] = np.random.randint(0.24*R//EPSILON, 0.30*R//EPSILON, size=(H,W))[random_mask==0] * EPSILON

import matplotlib.pyplot as plt
plt.imsave("ours.png",depth)

mask = np.zeros((H,W),dtype=int)+1
visualise_mask(depth, mask, INTRINSICS)

mask, planes = open3d_find_planes(depth, INTRINSICS, 0.002, CONFIDENCE, INLIER_THRESHOLD, MAX_PLANE, verbose=True)

print(mask.max())
print(planes)

visualise_mask(depth, mask, INTRINSICS)

information, mask, planes = plane_ransac(depth, INTRINSICS, R, EPSILON, SIGMA, CONFIDENCE, INLIER_THRESHOLD, MAX_PLANE, verbose=True)

print(information)
print(planes)

smallest = np.argmin(information)

mask[mask>smallest] = 0
print(mask.max())

visualise_mask(depth, mask, INTRINSICS)

