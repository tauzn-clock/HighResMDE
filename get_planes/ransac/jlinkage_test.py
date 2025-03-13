import json
import os
from PIL import Image
import numpy as np
import open3d as o3d
from depth_to_pcd import depth_to_pcd
from tqdm import tqdm

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
depth = depth[::8, ::8]
INTRINSICS[0] /= 2
INTRINSICS[5] /= 2
INTRINSICS[2] /= 2
INTRINSICS[6] /= 2

H, W = depth.shape

print(H, W)

pts, idx = depth_to_pcd(depth, INTRINSICS)

mss = np.zeros((len(pts), MODEL_CNT), dtype=np.bool)

for i in tqdm(range(MODEL_CNT)):
    # Randomly select 3 points
    p1, p2, p3 = np.random.choice(len(pts), 3, replace=False)
    # Calculate the plane
    v1 = pts[p2] - pts[p1]
    v2 = pts[p3] - pts[p1]
    normal = np.cross(v1, v2)
    normal = normal / np.linalg.norm(normal)
    d = -np.dot(normal, pts[p1])

    dist = np.abs(np.dot(pts, normal) + d)
    mss[:, i] = dist < THRESHOLD

def get_jaccard(a, b):
    union = np.logical_or(a, b).sum()
    intersection = np.logical_and(a, b).sum()
    return intersection / union

mask = np.array(range(H*W))
idx_mask = np.array(range(H*W))
mask = mask.reshape((H, W))
idx_mask = idx_mask.reshape((H, W))
print(mask.shape)

adj = [(0, 1), (0, -1), (1, 0), (-1, 0)]

for _ in range(len(pts)):
    best_jaccard = 0
    index_a = -1
    index_b = -1

    for i in range(mss.shape[0]):
        i_x = idx[i][1]
        i_y = idx[i][0]

        for coord in adj:
            j_x = i_x + coord[0]
            j_y = i_y + coord[1]

            if j_x < 0 or j_x >= W or j_y < 0 or j_y >= H:
                continue

            i = mask[i_y, i_x]
            j = mask[j_y, j_x]

            if i == j:
                continue

            jaccard = get_jaccard(mss[i,:], mss[j,:])
            if jaccard > best_jaccard:
                best_jaccard = jaccard
                index_a = i
                index_b = j

    print(f'Best jaccard: {best_jaccard}')
    if best_jaccard < 0.01:
        break

    mask[index[index_b]] = mask[index[index_a]]
    mss[index_a, :] = np.logical_and(mss[index_a, :], mss[index_b, :])
    mss = np.delete(mss, index_b, axis=0)
    index = np.delete(index, index_b)
    
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts)
o3d.visualization.draw_geometries([pcd])