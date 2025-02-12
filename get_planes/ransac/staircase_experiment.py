import numpy as np
from information_estimation import plane_ransac
from visualise import visualise_mask
from synthetic_test import set_depth, open3d_find_planes
H = 480
W = 640
R = 10
EPSILON = 0.001
SIGMA = np.ones(H*W) * 0.01
MAX_PLANE = 8
CONFIDENCE = 0.99
INLIER_THRESHOLD = 0.15
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

N = 4
distance = [1.63,1,1,1.63]
distance = np.array(distance) * -(0.5)**0.5
for i in range(N):
    mask = np.zeros((H,W))
    mask[i*H//N:(i+1)*H//N,:] = 1

    angle = -(-1)**i * np.pi/4

    normal = np.array([0,np.sin(angle), np.cos(angle)])
    #normal = [0,0,1]

    depth += set_depth(np.ones((H,W)),INTRINSICS, mask, normal, distance[i])


#Add noise
depth += np.random.normal(0,0.01,(H,W))

depth = np.array(depth/EPSILON,dtype=int) * EPSILON

import matplotlib.pyplot as plt
plt.imsave("staircase.png",depth)

mask, planes = open3d_find_planes(depth, INTRINSICS, 0.01, CONFIDENCE, INLIER_THRESHOLD, MAX_PLANE, verbose=True)
visualise_mask(depth, mask, INTRINSICS)

information, mask, planes = plane_ransac(depth, INTRINSICS, R, EPSILON, SIGMA, CONFIDENCE, INLIER_THRESHOLD, MAX_PLANE, verbose=True)
print(information)
print(planes)

smallest = np.argmin(information)

mask[mask>smallest] = 0
print(mask.max())

visualise_mask(depth, mask, INTRINSICS)
