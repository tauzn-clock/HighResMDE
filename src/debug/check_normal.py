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

from torchvision.transforms import GaussianBlur
from plane_estimation import normal_to_planes

import matplotlib.pyplot as plt

args = global_parser()
local_rank = "cpu"

train_dataset = BaseImageDataset('train', NYUImageData, args.test_dir, args.test_csv)
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True)

config = ModelConfig(args.model_size)
model = Model(config).to(local_rank)

model.eval()

dn_to_distance = DN_to_distance(args.batch_size, args.height, args.width).to(local_rank)
normal_estimation = Depth2Normal(local_rank).to(local_rank)

loop = tqdm.tqdm(train_dataloader, desc=f"Epoch {1}", unit="batch")
for itr, x in enumerate(loop):
    if itr<10: continue
    for k in x.keys():
        x[k] = x[k].to(local_rank)
    break

depth_gt = x["depth_values"] #Unit: m
normal_gt, x["mask"] = normal_estimation(depth_gt, x["camera_intrinsics"], x["mask"], 1.0) # Intrinsic needs to be in mm, ideally change depth_gt to mm for consistency, skip for speed
normal_gt = F.normalize(normal_gt, dim=1, p=2) #Unit: none, normalised
dist_gt = dn_to_distance(depth_gt, normal_gt, x["camera_intrinsics_inverted"]) #Camera intrinsic needs to be in mm, but dist_gt is in m, probably dont need to scale depth_gt but just to be safe


# Plane estimator

PLANE_CNT = 128
K_MEAN_ITERATION = 10
KERNEL_SIZE = 31

B, _, H, W = depth_gt.shape


gaussian_blur = GaussianBlur(kernel_size=KERNEL_SIZE, sigma=(KERNEL_SIZE-1)/6)
for b in range(B):
    normal_gt[b] = gaussian_blur(normal_gt[b])
normal_gt = normal_gt * x["mask"]
normal_gt = F.normalize(normal_gt, dim=1, p=2) #Unit: none, normalised
dist_gt = dn_to_distance(depth_gt, normal_gt, x["camera_intrinsics_inverted"]) #Camera intrinsic needs to be in mm, but dist_gt is in m, probably dont need to scale depth_gt but just to be safe

plane = normal_to_planes(normal_gt, dist_gt, x["mask"], PLANE_CNT, K_MEAN_ITERATION)

normal = normal_gt[1].cpu().numpy().transpose((1,2,0))
normal = (normal + 1)/2
dist = dist_gt[1].cpu().numpy().squeeze()
p = plane[1].cpu().numpy().squeeze()
print(p.max(), p.min())

plt.imsave("normal.png", normal)
plt.imsave("dist.png", dist)
plt.imsave("plane.png", p)