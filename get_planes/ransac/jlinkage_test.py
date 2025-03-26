import json
import os
from PIL import Image
import numpy as np
import open3d as o3d
from depth_to_pcd import depth_to_pcd
from tqdm import tqdm
from matplotlib import pyplot as plt

DIR_FILE = "/scratchdata/processed/alcove2"
MODEL_CNT = 1000
THRESHOLD = 0.001

with open(os.path.join(DIR_FILE, "camera_info.json"), "r") as f:
    camera_info = json.load(f)

INTRINSICS = camera_info["K"]
print(INTRINSICS)

INDEX = 0

depth = Image.open(os.path.join(DIR_FILE, "depth", f"{INDEX}.png"))
depth = np.array(depth)

#Downsample
RESCALE = 20
depth = depth[::RESCALE, ::RESCALE] / 1000
INTRINSICS[0] /= RESCALE
INTRINSICS[5] /= RESCALE
INTRINSICS[2] /= RESCALE
INTRINSICS[6] /= RESCALE

H, W = depth.shape

print(H, W)

# Create dummy depth

depth = np.zeros((H, W))
depth[:H//2, :W//2] = 2
depth[H//2:, :W//2] = 3
depth[:H//2, W//2:] = 4
depth[H//2:, W//2:] = 5

print(depth.max())

kernel = np.array([[1,2,4,2,1],
                   [2,4,8,4,2],
                   [4,8,0,8,4],
                   [2,4,8,4,2],
                   [1,2,4,2,1]])



pts, _ = depth_to_pcd(depth, INTRINSICS)

mss = np.zeros((len(pts), MODEL_CNT), dtype=np.bool)

for i in tqdm(range(MODEL_CNT)):

    while True:
        p1 = np.random.randint(0, len(pts))
        p1_x = p1 % W
        p1_y = p1 // W
        
        # Randomly select 2 points, centered around p1 with a kernel
        p2 = np.random.choice(len(kernel.flatten()), 1, p=kernel.flatten()/kernel.sum())[0]
        p2_x = p1_x + p2 % 5 - 2
        p2_y = p1_y + p2 // 5 - 2
        p2 = p2_y * W + p2_x

        p3 = np.random.choice(len(kernel.flatten()), 1, p=kernel.flatten()/kernel.sum())[0]
        p3_x = p1_x + p3 % 5 - 2
        p3_y = p1_y + p3 // 5 - 2
        p3 = p3_y * W + p3_x

        if p2 >= 0 and p2 < H*W and p3 >= 0 and p3 < H*W:
            if pts[p1, 2] > 0 and pts[p2, 2] > 0 and pts[p3, 2] > 0:
                v1 = pts[p2] - pts[p1]
                v2 = pts[p3] - pts[p1]
                normal = np.cross(v1, v2)
                if np.linalg.norm(normal) > 1e-10:
                    normal = normal / (np.linalg.norm(normal)+1e-10)
                    d = -np.dot(normal, pts[p1])
                    break

    dist = np.abs(np.dot(pts, normal) + d)
    mss[:, i] = dist < THRESHOLD

print(mss.shape)

plt.imsave('mss.png', mss, cmap='gray')

def get_jaccard(a, b):
    union = np.logical_or(a, b).sum()
    intersection = np.logical_and(a, b).sum()
    return intersection / union

parent = np.linspace(0, len(pts)-1, len(pts), dtype=np.int32)
index = np.ones(len(pts), dtype=bool)

parent[pts[:, 2] <= 0] = -1
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
            if parent[idx_j] == -1:
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
    print(best_a, best_b)
    if best_jaccard < 0.5:
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
print(features, counts)

test = parent.reshape((H, W))
plt.imsave('jlink_mss.png', test, cmap='gray')
new_mss = mss.copy()
for i in range(len(parent)):
    new_mss[i] = mss[parent[i], :]
plt.imsave('mss_new.png', new_mss, cmap='gray')

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts)
o3d.visualization.draw_geometries([pcd])