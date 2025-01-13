import sys
sys.path.append('/HighResMDE/src')

from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import tqdm
import csv
import os
from torchvision import transforms

from model import Model, ModelConfig
from dataloader.BaseDataloader import BaseImageDataset
from dataloader.NYUDataloader import NYUImageData
from layers.DN_to_distance import DN_to_distance
from layers.depth_to_normal import Depth2Normal
from loss import silog_loss, get_metrics
from segmentation import compute_seg, get_smooth_ND
from global_parser import global_parser
from infer_image import infer_image

import matplotlib.pyplot as plt

args = global_parser()
local_rank = "cuda"

train_dataset = BaseImageDataset('train', NYUImageData, args.test_dir, args.test_csv)
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True)

config = ModelConfig(args.model_size)
model = Model(config).to(local_rank)

model.eval()

dn_to_distance = DN_to_distance(args.batch_size, args.height, args.width).to(local_rank)
normal_estimation = Depth2Normal(local_rank).to(local_rank)

loop = tqdm.tqdm(train_dataloader, desc=f"Epoch {1}", unit="batch")
for itr, x in enumerate(loop):
    #if itr<2: continue
    for k in x.keys():
        x[k] = x[k].to(local_rank)
    break

depth_gt = x["depth_values"] #Unit: m
normal_gt, x["mask"] = normal_estimation(depth_gt, x["camera_intrinsics"], x["mask"], args.normal_blur) # Intrinsic needs to be in mm, ideally change depth_gt to mm for consistency, skip for speed
normal_gt = F.normalize(normal_gt, dim=1, p=2) #Unit: none, normalised
dist_gt = dn_to_distance(depth_gt, normal_gt, x["camera_intrinsics_inverted"]) #Camera intrinsic needs to be in mm, but dist_gt is in m, probably dont need to scale depth_gt but just to be safe

# Plane estimator

PLANE_CNT = 1024
K_MEAN_ITERATION = 100

B, _, H, W = depth_gt.shape

coords_h = torch.arange(H).to(local_rank)
coords_w = torch.arange(W).to(local_rank)
index_mesh = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))

plane = torch.randint(1, PLANE_CNT+1, x["depth_values"].shape).to(local_rank)
plane = plane * x["mask"]
for b in range(B):
    index_valid = index_mesh[:, x["mask"][b].squeeze()]
    index_select = index_valid[:, :PLANE_CNT]

    store_normal = normal_gt[b,:,index_select[0,:],index_select[1,:]]
    store_dist = dist_gt[b,:,index_select[0,:],index_select[1,:]]

    # Re Cluster
    normal_gt_flatten = normal_gt[b].view(3, -1)
    normal_distance_function = 1 - torch.matmul(store_normal.t(), normal_gt_flatten)
    normal_distance_function = normal_distance_function.view(PLANE_CNT, *normal_gt[b].shape[1:])

    dist_gt_flatten = dist_gt[b].view(-1)
    dist_distance_function = torch.abs(store_dist.t() - dist_gt_flatten)
    dist_distance_function = dist_distance_function.view(PLANE_CNT, *normal_gt[b].shape[1:])

    distance_function = normal_distance_function + dist_distance_function
    
    new_plane = torch.argmin(distance_function, dim=0) + 1

    plane[b] = new_plane * x["mask"][b]

for b in range(B):
    for em_itr in range(K_MEAN_ITERATION):
        store_normal = torch.zeros((3, PLANE_CNT)).to(local_rank)
        store_dist = torch.zeros(PLANE_CNT).to(local_rank)
        
        # Average
        for i in range(1, PLANE_CNT+1):
            mask = (plane[b]==i).float()
            if mask.sum() == 0: continue
            normal = normal_gt[b] * mask.unsqueeze(0)
            normal = normal.squeeze(0)
            normal_mean = normal.sum(dim=(1,2)) / mask.sum()
            dist = dist_gt[b] * mask
            dist_mean = dist.sum() / mask.sum()
            store_normal[:, i-1] = normal_mean
            store_dist[i-1] = dist_mean

        store_normal = F.normalize(store_normal, dim=0)
        
        # Re Cluster
        normal_gt_flatten = normal_gt[b].view(3, -1)
        normal_distance_function = 1 - torch.matmul(store_normal.t(), normal_gt_flatten)
        normal_distance_function = normal_distance_function.view(PLANE_CNT, *normal_gt[b].shape[1:])

        dist_gt_flatten = dist_gt[b].view(-1).unsqueeze(0)
        dist_distance_function = torch.abs(store_dist.unsqueeze(0).t() - dist_gt_flatten)
        dist_distance_function = dist_distance_function.view(PLANE_CNT, *normal_gt[b].shape[1:])

        distance_function = normal_distance_function + dist_distance_function

        new_plane = torch.argmin(distance_function, dim=0) + 1

        plane[b] = new_plane * x["mask"][b]


normal = normal_gt[0].cpu().numpy().transpose((1,2,0))
normal = (normal + 1)/2
dist = dist_gt[0].cpu().numpy().squeeze()
p = plane[0].cpu().numpy().squeeze()
print(p.max(), p.min())

plt.imsave("normal.png", normal)
plt.imsave("dist.png", dist)
plt.imsave("plane.png", p)