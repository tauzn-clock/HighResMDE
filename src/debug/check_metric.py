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

print("Using ", args.pretrained_model)
model.load_state_dict(torch.load(args.pretrained_model, weights_only=False))
torch. set_grad_enabled(False)
torch.cuda.empty_cache()

model.eval()

loop = tqdm.tqdm(train_dataloader, desc=f"Epoch {1}", unit="batch")
for itr, x in enumerate(loop):
    #if itr<84: continue
    for k in x.keys():
        x[k] = x[k].to(local_rank)

    pred = infer_image(model,x)[0]
    gt = x["depth_values"][0]
    mask = x["mask"][0]
    color = x["pixel_values"][0]

    thresh = torch.maximum((gt / pred), (pred / gt))
    tmp = (thresh[mask]<1.25).float().mean()
    print(tmp)
    if  tmp < 0.9:
        thresh[thresh>100] = 1
        print(thresh.max(), thresh.min())

        plt.imsave("thresh.png", thresh.cpu().detach().squeeze())
        plt.imsave("gt.png", gt.cpu().detach().squeeze())
        plt.imsave("pred.png", pred.cpu().detach().squeeze())
        plt.imsave("color.png",color.cpu().permute(1, 2, 0).numpy())
        break