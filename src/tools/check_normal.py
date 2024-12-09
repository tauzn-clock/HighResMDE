import sys
sys.path.append('/HighResMDE/src')

from PIL import Image
import torch
import torch.nn as nn
import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from layers.depth_to_normal import Depth2Normal
from dataloader.BaseDataloader import BaseImageDataset
from dataloader.NYUDataloader import NYUImageData
from layers.DN_to_distance import DN_to_distance
from layers.DN_to_depth import DN_to_depth
from torchvision import transforms
import matplotlib.pyplot as plt

BATCH_SIZE = 10
DEVICE = "cuda"
train_dataset = BaseImageDataset('train', NYUImageData, '/', '/HighResMDE/src/nyu_test_v2.csv')
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, pin_memory=True)

normal_estimation = Depth2Normal().to(DEVICE)
blur = transforms.GaussianBlur(kernel_size=5)
dn_to_distance = DN_to_distance(BATCH_SIZE, 480, 640).to(DEVICE)

loop = tqdm.tqdm(train_dataloader, unit="batch")
for itr, x in enumerate(loop):
    for k in x.keys():
        x[k] = x[k].to(DEVICE)
    depth_gt = x["depth_values"] #Unit: m
    normal_gt, x["mask"] = normal_estimation(depth_gt, x["camera_intrinsics_mm"], x["mask"], 5.0) # TODO: Scale add as blur
    #normal_gt = torch.stack([blur(each_normal) for each_normal in normal_gt])
    normal_gt = F.normalize(normal_gt, dim=1, p=2) #Unit: none, normalised
    dist_gt = dn_to_distance(depth_gt, normal_gt, x["camera_intrinsics_mm_inverted"]) #Unit: m

    # For testing
    b, _, h, w =  normal_gt.shape 
    device = normal_gt.device  
    dn_to_depth = DN_to_depth(b, h, w).to(device)
    depth_est = dn_to_depth(normal_gt, dist_gt, x["camera_intrinsics_mm_inverted"])#.clamp(0, 1)
    break

plt.imshow(x["pixel_values"][0].permute(1,2,0).cpu().numpy())
plt.savefig(f"normal_rgb.png", bbox_inches='tight', dpi=300)

plt.imshow(x["depth_values"][0].permute(1,2,0).cpu().numpy())
plt.savefig(f"normal_depth.png", bbox_inches='tight', dpi=300)
print("Normal Depth", x["depth_values"][0].max(), x["depth_values"][0].min())

plt.imshow((normal_gt[0].permute(1,2,0).cpu().numpy() + 1)/2)
plt.savefig(f"normal_gt.png", bbox_inches='tight', dpi=300)
print(normal_gt[0].max(), normal_gt[0].min())

plt.imshow(dist_gt[0].permute(1,2,0).cpu().numpy()/(dist_gt[0].cpu().numpy().max()))
plt.savefig(f"dist_gt.png", bbox_inches='tight', dpi=300)
print(dist_gt[0].max(), dist_gt[0].min())

plt.imshow(depth_est[0].permute(1,2,0).cpu().numpy()/(depth_est[0].cpu().numpy().max()))
plt.savefig(f"depth_est.png", bbox_inches='tight', dpi=300)
print(depth_est[0].max(), depth_est[0].min())

plt.imshow(x["mask"][0].cpu().numpy().squeeze())
plt.savefig(f"mask.png", bbox_inches='tight', dpi=300)