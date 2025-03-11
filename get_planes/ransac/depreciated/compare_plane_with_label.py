import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import csv
from metrics import plane_ordering
from depth_to_pcd import depth_to_pcd
from PIL import Image

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
FOLDER="new_gt_noise_model"
data_csv = "/HighResMDE/get_planes/ransac/config/nyu.csv"
with open(data_csv, 'r') as f:
    reader = csv.reader(f)
    DATA = list(reader)

metric = []

for INDEX in range(1449):
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
    SIGMA = 0.0012 + 0.0019 * (points[:,2] - 0.4)**2
    pred, pred_csv = plane_ordering(points, pred.flatten(), pred_csv, R, EPSILON, SIGMA, merge_planes=True)
    pred = pred.reshape(depth.shape)

    #pred = cv2.imread(os.path.join(ROOT, "original_gt", f"{INDEX}.png"), cv2.IMREAD_UNCHANGED)

    each_metrics = []

    for key in flat_labels:
        gt_mask = gt == flat_labels[key]

        if gt_mask.sum() == 0: 
            each_metrics.append([0, 0, 0, 0])
            continue

        best_iou = 0
        best_iou_index = -1

        for i in range(1, pred.max()+1):
            pred_mask = pred == i

            # Calculate IoU
            intersection = np.logical_and(gt_mask, pred_mask)
            union = np.logical_or(gt_mask, pred_mask)

            iou = intersection.sum() / union.sum()
            
            if iou > best_iou:
                best_iou = iou
                best_iou_index = i

        if best_iou_index == -1: 
            each_metrics.append([0, 0, 0, 0])
        else:
            each_metrics.append([best_iou] + get_metric(gt_mask, pred == best_iou_index))
        
    each_metrics = np.array(each_metrics)
    metric.append(each_metrics)

metric = np.array(metric)
print(metric.shape)

print(np.mean(metric, axis=0))

"""
Original GT:
[[0.49364788 0.79220085 0.56875876 0.62509793]
 [0.06407709 0.08531529 0.06755748 0.07279495]
 [0.62487065 0.70743184 0.66357643 0.67273705]
 [0.1238822  0.16984607 0.15566408 0.14933553]
 [0.10548322 0.17097035 0.13682334 0.13522535]]
"""

"""
Proportional Noise:
[[0.48744679 0.6825865  0.64955511 0.62598243]
 [0.0630987  0.08932275 0.0859664  0.07679088]
 [0.56796911 0.65939146 0.67573827 0.64307086]
 [0.10498988 0.14727822 0.17113001 0.13481629]
 [0.05238704 0.12023923 0.09437795 0.07577731]]
Specific noise model:
[[0.49123787 0.72438183 0.61991709 0.63132005]
 [0.06303651 0.08938695 0.08253578 0.0774748 ]
 [0.57124625 0.69272072 0.65317563 0.65177171]
 [0.10990859 0.15927535 0.1617148  0.14005832]
 [0.0580108  0.14264616 0.08385726 0.08200619]]
"""