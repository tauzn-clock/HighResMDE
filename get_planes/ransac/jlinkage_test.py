import json
import os
from PIL import Image
import numpy as np
import open3d as o3d
from depth_to_pcd import depth_to_pcd
from tqdm import tqdm
from matplotlib import pyplot as plt

DIR_FILE = "/scratchdata/processed/alcove2"
MODEL_CNT = 100000
THRESHOLD = 0.1

with open(os.path.join(DIR_FILE, "camera_info.json"), "r") as f:
    camera_info = json.load(f)

INTRINSICS = camera_info["K"]
print(INTRINSICS)

INDEX = 0

depth = Image.open(os.path.join(DIR_FILE, "depth", f"{INDEX}.png"))
depth = np.array(depth)

#Downsample
depth = depth[::10, ::10]
INTRINSICS[0] /= 2
INTRINSICS[5] /= 2
INTRINSICS[2] /= 2
INTRINSICS[6] /= 2

H, W = depth.shape

print(H, W)

pts, _ = depth_to_pcd(depth, INTRINSICS)
for_sampling = pts[pts[:, 2] > 0]

mss = np.zeros((len(pts), MODEL_CNT), dtype=np.bool)

for i in tqdm(range(MODEL_CNT)):
    # Randomly select 3 points
    p1, p2, p3 = np.random.choice(len(for_sampling), 3, replace=False)
    # Calculate the plane
    v1 = for_sampling[p2] - for_sampling[p1]
    v2 = for_sampling[p3] - for_sampling[p1]
    normal = np.cross(v1, v2)
    normal = normal / np.linalg.norm(normal)
    d = -np.dot(normal, pts[p1])

    dist = np.abs(np.dot(pts, normal) + d)
    mss[:, i] = dist < THRESHOLD

def get_jaccard(a, b):
    union = np.logical_or(a, b).sum()
    intersection = np.logical_and(a, b).sum()
    return intersection / union

parent = np.linspace(0, len(pts)-1, len(pts), dtype=np.int32)
index = np.ones(len(pts), dtype=bool)

parent[pts[:, 2] <= 0] = 0
index[pts[:, 2] <= 0] = False

adj = [(0, 1), (0, -1), (1, 0), (-1, 0)]

for itr in range(len(pts)):
    best_jaccard = 0
    index_a = -1
    index_b = -1

    for idx_i, i in enumerate(index):
        if not i:
            continue

        i_x = idx_i % W 
        i_y = idx_i // W

        for coord in adj:
            j_x = i_x + coord[0]
            j_y = i_y + coord[1]

            if j_x < 0 or j_x >= W or j_y < 0 or j_y >= H:
                continue

            idx_j = j_y * W + j_x
            if pts[idx_j, 2] <= 0:
                continue
            
            idx_j = parent[idx_j]
            while parent[idx_j] != idx_j:
                idx_j = parent[idx_j]
            
            if idx_j == idx_i:
                continue

            jaccard = get_jaccard(mss[parent[idx_i],:], mss[parent[idx_j],:])
            if jaccard > best_jaccard:
                best_jaccard = jaccard
                best_a = idx_i
                best_b = idx_j

    print(f'Best jaccard: {best_jaccard}', itr)
    if best_jaccard < 0.0001:
        break

    parent[best_b] = parent[best_a]
    mss[parent[best_a], :] = np.logical_and(mss[parent[best_a], :], mss[parent[best_b], :])
    index[best_b] = False

for i in range(parent.shape[0]):
    cur = parent[i]
    while parent[cur] != cur:
        cur = parent[cur]
    parent[i] = cur

#Find unique planes
features, counts = np.unique(parent, return_counts=True)
print(len(features))

test = parent.reshape((H, W))
plt.imsave('jlink_mss.png', test, cmap='gray')

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts)
o3d.visualization.draw_geometries([pcd])