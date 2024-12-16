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

import matplotlib.pyplot as plt

args = global_parser()
local_rank = "cuda"

train_dataset = BaseImageDataset('test', NYUImageData, args.train_dir, args.train_csv)
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True)

config = ModelConfig(args.model_size)
if not args.swinv2_specific_path is None: config.swinv2_pretrained_path = args.swinv2_specific_path
model = Model(config).to(local_rank)

print("Using ", args.pretrained_model)
model.load_state_dict(torch.load(args.pretrained_model, weights_only=False))
torch.cuda.empty_cache()
#model.backbone.backbone.from_pretrained(model.config.swinv2_pretrained_path)
# Freeze the encoder layers only
for param in model.backbone.backbone.parameters():  # 'backbone' is typically where the encoder layers reside
    param.requires_grad = False
#torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

silog_criterion = silog_loss(variance_focus=args.var_focus).to(local_rank)
dn_to_distance = DN_to_distance(args.batch_size, args.height, args.width).to(local_rank)
normal_estimation = Depth2Normal().to(local_rank)

model.eval()

loop = tqdm.tqdm(train_dataloader, desc=f"Epoch {1}", unit="batch")
for itr, x in enumerate(loop):
    break
for k in x.keys():
    x[k] = x[k].to(local_rank)
d1_list, u1, d2_list, u2, norm_est, dist_est = model(x)

# Estimate GT normal and distance

depth_gt = x["depth_values"] #Unit: m
normal_gt, x["mask"] = normal_estimation(depth_gt, x["camera_intrinsics"], x["mask"], args.normal_blur) # Intrinsic needs to be in mm, ideally change depth_gt to mm for consistency, skip for speed
#normal_gt = torch.stack([blur(each_normal) for each_normal in normal_gt])
normal_gt = F.normalize(normal_gt, dim=1, p=2) #Unit: none, normalised
dist_gt = dn_to_distance(depth_gt, normal_gt, x["camera_intrinsics_inverted"]) #Camera intrinsic needs to be in mm, but dist_gt is in m, probably dont need to scale depth_gt but just to be safe

for i in range(len(d1_list)):
    plt.imshow((d1_list[i][0]+d2_list[i][0]).cpu().detach().squeeze())
    plt.savefig(f"output_image_{i}.png", bbox_inches='tight', dpi=300)

norm_est_img = (norm_est[0] + 1)/2
print(norm_est_img.shape)
print(norm_est.max(), norm_est.min())
plt.imshow(norm_est_img.permute(1,2,0).cpu().detach().squeeze())
plt.savefig('normal_est.png')

dist_est_img = dist_est[0]/dist_est[0].max()
plt.imshow(dist_est_img.permute(1,2,0).cpu().detach().squeeze())
plt.savefig('dist_est.png')

normal_gt_img = (normal_gt[0] +1)/2
print(normal_gt_img.shape)
print(normal_gt.max(), normal_gt.min())
plt.imshow(normal_gt_img.permute(1,2,0).cpu().detach().squeeze())
plt.savefig('normal_gt.png')