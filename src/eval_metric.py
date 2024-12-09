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

args = global_parser()
local_rank = "cuda"

test_dataset = BaseImageDataset('test', NYUImageData, args.test_dir, args.test_csv)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, pin_memory=True)

config = ModelConfig(args.model_size)
if not args.swinv2_specific_path is None: config.swinv2_pretrained_path = args.swinv2_specific_path
model = Model(config).to(local_rank)

model.load_state_dict(torch.load(args.pretrained_model, weights_only=False))
torch.cuda.empty_cache()
model.backbone.backbone.from_pretrained(model.config.swinv2_pretrained_path)
# Freeze the encoder layers only
for param in model.backbone.backbone.parameters():  # 'backbone' is typically where the encoder layers reside
    param.requires_grad = False
#torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

model.eval()
torch.cuda.empty_cache()
tot_metric = [0 for _ in range(args.metric_cnt)]
cnt = 0
with torch.no_grad():
    for _, x in enumerate(tqdm.tqdm(test_dataloader)):
        for k in x.keys():
            x[k] = x[k].to(local_rank)
            
        d1_list, _, d2_list, _, _, _ = model(x)
        
        depth_gt = x["depth_values"]
        d1 = d1_list[-1]
        d2 = d2_list[-1]
        
        for b in range(x["max_depth"].shape[0]):
            metric = get_metrics(depth_gt[b], ((d1 + d2)/2)[b], x["mask"][b])
            assert len(metric) == args.metric_cnt
            for i in range(args.metric_cnt): tot_metric[i] += metric[i].cpu().detach().item()
            cnt+=1

for i in range(args.metric_cnt): tot_metric[i]/=cnt
print(tot_metric)