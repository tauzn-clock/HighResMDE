import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

DIR_PATH = "/home/daoxin/scratchdata/processed/outdoor"
INDEX = 10

img = cv2.imread(os.path.join(DIR_PATH, "rgb", f"{INDEX}.png"))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
depth = cv2.imread(os.path.join(DIR_PATH, "depth", f"{INDEX}.png"), cv2.IMREAD_UNCHANGED)
depth = depth.astype(np.float32) / 1000.0  # Convert to meters

# Canny edge detection
canny_edges = cv2.Canny(gray, 100, 400)
canny_edges = cv2.morphologyEx(canny_edges, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
print(canny_edges.shape, canny_edges.max(), canny_edges.min())
plt.imsave("canny_edges.png", canny_edges)

edges_inv = cv2.bitwise_not(canny_edges)

num_labels, labels = cv2.connectedComponents(edges_inv, connectivity=4)
print(num_labels, labels.shape, labels.max(), labels.min())

# Pixels with labels 0 are filled with label of closest pixel
segment_mask = labels.copy()
zero_indices = np.argwhere(segment_mask == 0)
neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1),
             (-1, -1), (-1, 1), (1, -1), (1, 1)]
while len(zero_indices)!=0:
    new_zero_indices = []
    new_segment_mask = segment_mask.copy()
    for i, j in zero_indices:
        # Get the 8-connected neighbors
        filled = False
        for ni, nj in neighbors:
            index_i = i + ni
            index_j = j + nj
            if 0 <= index_i < segment_mask.shape[0] and 0 <= index_j < segment_mask.shape[1]:
                if segment_mask[index_i, index_j] > 0:
                    new_segment_mask[i, j] = segment_mask[index_i, index_j]
                    filled = True
                    break
        if not filled:
            new_zero_indices.append((i, j))
    zero_indices = np.array(new_zero_indices) 
    segment_mask = new_segment_mask.copy()
    print(len(zero_indices))

plt.imsave("connected_components.png", segment_mask, cmap='hsv')