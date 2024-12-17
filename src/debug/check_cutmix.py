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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torchvision import transforms

from model import Model, ModelConfig
from dataloader.BaseDataloader import BaseImageDataset
from dataloader.NYUDataloader import NYUImageData
from layers.DN_to_distance import DN_to_distance
from layers.depth_to_normal import Depth2Normal
from loss import silog_loss, get_metrics
from segmentation import compute_seg, get_smooth_ND
from global_parser import global_parser

import matplotlib.pyplot as plt
from CutMix import CutMix

args = global_parser()
local_rank = "cuda"

train_dataset = BaseImageDataset('train', NYUImageData, args.train_dir, args.train_csv)
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True)

normal_estimation = Depth2Normal().to(local_rank)

loop = tqdm.tqdm(train_dataloader, unit="batch")
for itr, x in enumerate(loop):
    for k in x.keys():
        x[k] = x[k].to(local_rank)
    
    depth_gt = x["depth_values"] #Unit: m
    normal_gt, x["mask"] = normal_estimation(depth_gt, x["camera_intrinsics"], x["mask"], args.normal_blur) # Intrinsic needs to be in mm, ideally change depth_gt to mm for consistency, skip for speed
    normal_gt = F.normalize(normal_gt, dim=1, p=2) #Unit: none, normalised

    break

x["pixel_values"], x["depth_values"], x["mask"], normal_gt = CutMix(x["pixel_values"], x["depth_values"], x["mask"], normal_gt)

img = x["pixel_values"][0].cpu().numpy().transpose((1,2,0))
depth = x["depth_values"][0].cpu().numpy().squeeze()
normal = normal_gt[0].cpu().numpy().transpose((1,2,0))
mask = x["mask"][0].cpu().numpy().squeeze()
normal = (normal + 1) / 2

plt.imsave('img.png', img)
plt.imsave('mask.png', mask, cmap='jet')
plt.imsave('depth.png', depth, cmap='jet')
plt.imsave('normal.png', normal)