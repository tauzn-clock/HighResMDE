import numpy as np
from information_estimation import plane_ransac
from visualise import visualise_mask
from synthetic_test import set_depth, open3d_find_planes
import open3d as o3d
import json

dataset = o3d.data.SampleRedwoodRGBDImages()

print(len(dataset.depth_paths))

rgbd_images = []
for i in range(3,4):
    color_raw = o3d.io.read_image(dataset.color_paths[i])
    depth_raw = o3d.io.read_image(dataset.depth_paths[i])
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                                               color_raw, depth_raw)
    rgbd_images.append(rgbd_image)

with open(dataset.camera_intrinsic_path) as json_file:
    json_file = json.load(json_file)

H = int(json_file["height"])
W = int(json_file["width"])
INTRINSICS = [json_file["intrinsic_matrix"][0], 0, json_file["intrinsic_matrix"][6], 0, 0, json_file["intrinsic_matrix"][4], json_file["intrinsic_matrix"][7]]
print(INTRINSICS)

depth = np.array(depth_raw)
H, W = depth.shape

pcd = o3d.io.read_point_cloud(dataset.reconstruction_path)

o3d.visualization.draw_geometries([pcd])

R = 10000
EPSILON = 1
SIGMA = np.ones(H*W) * 10
MAX_PLANE = 10
CONFIDENCE = 0.99
INLIER_THRESHOLD = 0.2

mask, planes = open3d_find_planes(depth, INTRINSICS, 10, CONFIDENCE, INLIER_THRESHOLD, MAX_PLANE, verbose=True)

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

