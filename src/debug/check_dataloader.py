import sys
sys.path.append('/HighResMDE/src')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tqdm

from dataloader.BaseDataloader import BaseImageDataset
from dataloader.NYUDataloader import NYUImageData

from global_parser import global_parser

import matplotlib.pyplot as plt

args = global_parser()
local_rank = "cuda"

train_dataset = BaseImageDataset('train', NYUImageData, args.train_dir, args.train_csv)
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True)

loop = tqdm.tqdm(train_dataloader, desc=f"Epoch {1}", unit="batch")
for itr, x in enumerate(loop):
    if itr==10: break
for k in x.keys():
    x[k] = x[k].to(local_rank)

img = x["pixel_values"][0].permute(1,2,0).cpu().numpy()
plt.imsave("img.png", img)
depth = x["depth_values"][0].cpu().detach().squeeze()
plt.imsave("depth.png", depth)