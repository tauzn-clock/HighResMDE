import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import csv
from metrics import plane_ordering
from depth_to_pcd import depth_to_pcd
from scipy.ndimage import label
from PIL import Image
import tqdm

def largest_connected_region(mask):
    # Label the connected components in the mask
    labeled_mask, num_features = label(mask)
    
    # If no regions are found, return None or an empty mask
    if num_features == 0:
        return None
    
    # Calculate the size of each region
    region_sizes = [np.sum(labeled_mask == i) for i in range(1, num_features + 1)]
    
    # Find the index of the largest region
    largest_region_idx = np.argmax(region_sizes) + 1
    
    # Create a mask for the largest region
    largest_region_mask = (labeled_mask == largest_region_idx)
    
    return largest_region_mask

def get_metric(gt, pred):
    precision = (pred & gt).sum() / pred.sum()
    recall = (pred & gt).sum() / gt.sum()
    f1 = 2 * precision * recall / (precision + recall)

    return [precision, recall, f1]

flat_labels = {}
flat_labels["wall"] = 1
flat_labels["ceiling"] = 22
flat_labels["floor"] = 2
flat_labels["door"] = 8
flat_labels["window"] = 9

ROOT="/scratchdata/nyu_plane"
#FOLDER="new_gt_noise_model"
FOLDER="new_gt_sigma_1"
data_csv = "/HighResMDE/get_planes/ransac/config/nyu.csv"
with open(data_csv, 'r') as f:
    reader = csv.reader(f)
    DATA = list(reader)

metric = []

for INDEX in tqdm.tqdm(range(1449)):
    data = DATA[INDEX]
    EPSILON = 1/float(data[6]) # Resolution
    R = float(data[7]) # Maximum Range

    INTRINSICS = np.array([float(data[2]), 0, float(data[4]), 0, 0, float(data[3]), float(data[5]), 0]) # fx, fy, cx, cy
    rgb = np.array(Image.open(os.path.join(ROOT, data[0]))) / 255.0
    depth = np.array(Image.open(os.path.join(ROOT, data[1])))/float(data[6])

    gt = cv2.imread(os.path.join(ROOT, "labels", f"{INDEX}.png"), cv2.IMREAD_UNCHANGED)

    pred = cv2.imread(f"{ROOT}/{FOLDER}/{INDEX}.png", cv2.IMREAD_UNCHANGED)
    with open(f"{ROOT}/{FOLDER}/{INDEX}.csv", 'r') as f:
        reader = csv.reader(f)
        pred_csv = np.array(list(reader), dtype=np.float32)

    points, index = depth_to_pcd(depth, INTRINSICS)
    SIGMA = 0.01 * points[:,2]
    #SIGMA = 0.0012 + 0.0019 * (points[:,2] - 0.4)**2
    pred, pred_csv = plane_ordering(points, pred.flatten(), pred_csv, R, EPSILON, SIGMA, merge_planes=True)
    pred = pred.reshape(depth.shape)

    #pred = cv2.imread(os.path.join(ROOT, "original_gt", f"{INDEX}.png"), cv2.IMREAD_UNCHANGED)

    each_metrics = []

    for key in flat_labels:
        gt_mask = gt == flat_labels[key]

        if gt_mask.sum() == 0: 
            each_metrics.append([0, 0, 0])
            continue
        
        gt_mask = largest_connected_region(gt_mask)

        pred_mask = pred[gt_mask]
        unique_elements, counts = np.unique(pred_mask, return_counts=True)
        if unique_elements[0] == 0:
            unique_elements = unique_elements[1:]
            counts = counts[1:]
        #print(unique_elements, counts)

        if len(unique_elements) == 0:
            each_metrics.append([0, 1, 0])
            continue
        
        each_metrics.append([counts.sum(), counts.max(), counts.max()/counts.sum()])

        """
        fig, ax = plt.subplots()
        ax.imshow(pred)
        ax.imshow(gt_mask, alpha=0.5, cmap='binary')
        ax.axis('off')
        plt.savefig(f"{key}_pred_mask.png", bbox_inches='tight', pad_inches=0, transparent=True)
        """
            
    each_metrics = np.array(each_metrics)
    metric.append(each_metrics)

metric = np.array(metric)
print(metric.shape)

np.save("sigma_1.npy", metric)

print(np.mean(metric, axis=0))

# Find percentage of images
print((metric[:,:,1]>0).sum(axis=0)/metric.shape[0])

# Find mean within the images that have at least one plane
print(flat_labels.keys())
for i in range(len(flat_labels)):    
    tmp = metric[metric[:,i,1]>0]
    tmp = tmp[:,i]
    print(tmp[:,2].mean())
#    print(np.mean(tmp, axis=0))