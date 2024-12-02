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

torch.manual_seed(42)

local_rank="cuda"

BATCH_SIZE = 8
MODEL_PATH = "/HighResMDE/src/model.pth"

test_dataset = BaseImageDataset('test', NYUImageData, '/scratchdata/nyu_data/', '/HighResMDE/src/nyu_test.csv')
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, pin_memory=True)

config =  ModelConfig("tiny07")
config.batch_size = BATCH_SIZE
config.height = 480//4
config.width = 640//4
model = Model(config).to(local_rank)
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))

model.eval()
torch.cuda.empty_cache()
METRIC_CNT = 9
tot_metric = [0 for _ in range(METRIC_CNT)]
cnt = 0
with torch.no_grad():
    for _, x in enumerate(tqdm.tqdm(test_dataloader)):
        for k in x.keys():
            x[k] = x[k].to(local_rank)
            
        d1_list, _, d2_list, _, _, _ = model(x)
        
        depth_gt = x["depth_values"] #* x["max_depth"].view(-1, 1, 1, 1)
        d1 = d1_list[-1]
        d2 = d2_list[-1]
        
        for b in range(x["max_depth"].shape[0]):
            metric = get_metrics(depth_gt[b], ((d1 + d2)/2)[b], x["mask"][b])
            assert len(metric) == METRIC_CNT
            for i in range(METRIC_CNT): tot_metric[i] += metric[i].cpu().detach().item()
            cnt+=1

for i in range(METRIC_CNT): tot_metric[i]/=cnt

print(tot_metric)