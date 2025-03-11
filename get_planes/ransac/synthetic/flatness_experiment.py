import sys
sys.path.append("/HighResMDE/get_planes/ransac")

import numpy as np
from information_estimation import plane_ransac
from visualise import visualise_mask, save_mask
from synthetic_test import set_depth, open3d_find_planes
import matplotlib.pyplot as plt
import open3d as o3d
from metrics import plane_ordering
from depth_to_pcd import depth_to_pcd

NOISE_LEVEL = 10

ROOT = "/HighResMDE/get_planes/flatness"
H = 480
W = 640
R = 10
EPSILON = 0.001
SIGMA = np.ones(H*W) * EPSILON * NOISE_LEVEL
MAX_PLANE = 8
CONFIDENCE = 0.99
INLIER_THRESHOLD = 0.25

INTRINSICS = np.array([500, 0, 320, 0, 0, 500, 240])
N = 4
depth = np.zeros((H,W))
mask = np.zeros((H,W),dtype=int)
noise_level = [0,2,10,100]
for i in range(N):
    plane_mask = np.zeros((H,W),dtype=bool)
    plane_mask[:,i*W//N:(i+1)*W//N] = True
    depth += set_depth(np.ones((H,W)),INTRINSICS, plane_mask, [0,0,1], -0.2 * i - 2.5)
    mask[plane_mask] = i+1

    if i!=0:
        amplitude = NOISE_LEVEL * EPSILON
        x = np.linspace(0,1,H)
        y = np.sin(2*np.pi*x*noise_level[i]) * amplitude
        noise = np.tile(y,(W,1)).T
        print(noise.shape)
        #noise = np.random.normal(0,EPSILON*25,(H,W))
        #noise_mask = (random_mask < noise_level[i]) & plane_mask
        depth[plane_mask] += noise[plane_mask]

depth = np.array(depth/EPSILON,dtype=int) * EPSILON
print(depth.max(), depth.min())

import matplotlib.pyplot as plt
plt.imsave(f"{ROOT}/our.png",depth,cmap='gray')
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
planes = planes[1:smallest+1]
print(mask.max())

points, index = depth_to_pcd(depth, INTRINSICS)
mask, planes = plane_ordering(points, mask, planes, R, EPSILON, SIGMA)
print(mask.max())   
print(planes)
#print(planes)
#print(mask.reshape(H,W)[0,0])
#print(mask.reshape(H,W)[0,160])
#print(mask.reshape(H,W)[0,320])
#print(mask.reshape(H,W)[0,480])

save_mask(mask.reshape(H,W), f"{ROOT}/{NOISE_LEVEL}_ours_our.png")
visualise_mask(depth, mask, INTRINSICS, filepath=f"{ROOT}/{NOISE_LEVEL}_ours_pcd_our.png")

