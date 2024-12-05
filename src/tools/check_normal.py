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
from torchvision import transforms
import matplotlib.pyplot as plt

BATCH_SIZE = 4
DEVICE = "cuda"
train_dataset = BaseImageDataset('test', NYUImageData, '/scratchdata/nyu_data/', '/HighResMDE/src/nyu_test.csv')
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, pin_memory=True)

normal_estimation = Depth2Normal().to(DEVICE)
blur = transforms.GaussianBlur(kernel_size=5)
dn_to_distance = DN_to_distance(BATCH_SIZE, 480, 640).to(DEVICE)

loop = tqdm.tqdm(train_dataloader, unit="batch")
for itr, x in enumerate(loop):
    for k in x.keys():
        x[k] = x[k].to(DEVICE)
    depth_gt = x["depth_values"] #Unit: m
    normal_gt, x["mask"] = normal_estimation(depth_gt * 1000, x["camera_intrinsics_mm"], x["mask"], 1.0) # TODO: Figure out what scale does
    #normal_gt = torch.stack([blur(each_normal) for each_normal in normal_gt])
    normal_gt = F.normalize(normal_gt, dim=1, p=2) #Unit: none, normalised
    dist_gt = dn_to_distance(depth_gt, normal_gt, x["camera_intrinsics_mm_inverted"]) #Unit: m
    break

plt.imshow(x["pixel_values"][0].permute(1,2,0).cpu().numpy())
plt.savefig(f"normal_rgb.png", bbox_inches='tight', dpi=300)

plt.imshow(x["depth_values"][0].permute(1,2,0).cpu().numpy())
plt.savefig(f"normal_depth.png", bbox_inches='tight', dpi=300)

plt.imshow((normal_gt[0].permute(1,2,0).cpu().numpy() + 1)/2)
plt.savefig(f"normal_gt.png", bbox_inches='tight', dpi=300)
print(normal_gt[0].max(), normal_gt[0].min())

plt.imshow(dist_gt[0].permute(1,2,0).cpu().numpy()/(dist_gt[0].cpu().numpy().max()))
plt.savefig(f"dist_gt.png", bbox_inches='tight', dpi=300)
print(dist_gt[0].max(), dist_gt[0].min())