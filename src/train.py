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

def init_process_group(local_rank, world_size):
    dist.init_process_group(
        backend='nccl',  # Use NCCL backend for multi-GPU communication
        rank=local_rank,
        world_size=world_size
    )
    torch.cuda.set_device(local_rank)  # Set the GPU device for the current process

def main(local_rank, world_size):
    init_process_group(local_rank, world_size)
    
    BATCH_SIZE = 10
    MODEL_SIZE = "large07"
    SWINV2_SPECIFIC_PATH = None #"microsoft/swinv2-tiny-patch4-window8-256"
    VAR_FOCUS = 0.85
    LR = 2e-4
    LR_DECAY = 0.975

    LOSS_DEPTH_WEIGHT = 1
    LOSS_UNCER_WEIGHT = 1
    LOSS_NORMAL_WEIGHT = 5
    LOSS_DIST_WEIGHT = 0.25

    METRIC_CNT = 9
    
    train_dataset = BaseImageDataset('train', NYUImageData, '/scratchdata/nyu_data/', '/HighResMDE/src/nyu_train.csv')
    train_sampler = torch.utils.data.DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, pin_memory=True, sampler=train_sampler)

    test_dataset = BaseImageDataset('test', NYUImageData, '/scratchdata/nyu_data/', '/HighResMDE/src/nyu_test.csv')
    test_sampler = torch.utils.data.DistributedSampler(test_dataset, num_replicas=world_size, rank=local_rank)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, pin_memory=True, sampler=test_sampler)

    csv_file = [["silog", "abs_rel", "log10", "rms", "sq_rel", "log_rms", "d1", "d2", "d3"]]
    with open('metric.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(csv_file)

    config =  ModelConfig(MODEL_SIZE)
    config.batch_size = BATCH_SIZE
    config.height = 480//4
    config.width = 640//4
    if not SWINV2_SPECIFIC_PATH is None: config.swinv2_pretrained_path = SWINV2_SPECIFIC_PATH
    model = Model(config).to(local_rank)
    model.backbone.backbone.from_pretrained(model.config.swinv2_pretrained_path)
    # Freeze the encoder layers only
    for param in model.backbone.backbone.parameters():  # 'backbone' is typically where the encoder layers reside
        param.requires_grad = False
    #torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    silog_criterion = silog_loss(variance_focus=VAR_FOCUS).to(local_rank)
    dn_to_distance = DN_to_distance(config.batch_size, config.height * 4, config.width * 4).to(local_rank)
    normal_estimation = Depth2Normal().to(local_rank)
    blur = transforms.GaussianBlur(kernel_size=5)

    for epoch in range(50):
        model.train()

        loop = tqdm.tqdm(train_dataloader, desc=f"Epoch {epoch+1}", unit="batch")
        for itr, x in enumerate(loop):
            optimizer.zero_grad()
            for k in x.keys():
                x[k] = x[k].to(local_rank)
            if x["depth_values"].shape[0] != BATCH_SIZE: continue # Hacky solution to deal with batch size issue

            d1_list, u1, d2_list, u2, norm_est, dist_est = model(x)

            #print(d1_list[0].max())
            
            # Estimate GT normal and distance

            depth_gt = x["depth_values"] #Unit: m
            normal_gt, x["mask"] = normal_estimation(depth_gt, x["camera_intrinsics_mm"], x["mask"], 1.0) # Intrinsic needs to be in mm, ideally change depth_gt to mm for consistency, skip for speed
            #normal_gt = torch.stack([blur(each_normal) for each_normal in normal_gt])
            normal_gt = F.normalize(normal_gt, dim=1, p=2) #Unit: none, normalised
            dist_gt = dn_to_distance(depth_gt, normal_gt, x["camera_intrinsics_mm_inverted"]) #Camera intrinsic needs to be in mm, but dist_gt is in m, probably dont need to scale depth_gt but just to be safe
            
            # Depth Loss

            loss_depth1_0 = silog_criterion(d1_list[0], depth_gt, x["mask"])
            loss_depth2_0 = silog_criterion(d2_list[0], depth_gt, x["mask"])

            loss_depth1 = 0
            loss_depth2 = 0
            weights_sum = 0
            for i in range(len(d1_list) - 1):
                loss_depth1 += (VAR_FOCUS**(len(d1_list)-i-2)) * silog_criterion(d1_list[i + 1], depth_gt, x["mask"])
                loss_depth2 += (VAR_FOCUS**(len(d2_list)-i-2)) * silog_criterion(d2_list[i + 1], depth_gt, x["mask"])
                weights_sum += VAR_FOCUS**(len(d1_list)-i-2)
            
            loss_depth =  ((loss_depth1 + loss_depth2) / weights_sum + loss_depth1_0 + loss_depth2_0 )
            
            # Uncertainty Loss

            uncer1_gt = torch.exp(-5 * torch.abs(depth_gt - d1_list[0].detach()) / (depth_gt + d1_list[0].detach() + 1e-7))
            uncer2_gt = torch.exp(-5 * torch.abs(depth_gt - d2_list[0].detach()) / (depth_gt + d2_list[0].detach() + 1e-7))
            
            loss_uncer1 = torch.abs(u1-uncer1_gt)[x["mask"]].mean()
            loss_uncer2 = torch.abs(u2-uncer2_gt)[x["mask"]].mean()

            loss_uncer =  (loss_uncer1 + loss_uncer2)

            loss_normal = LOSS_NORMAL_WEIGHT * (1 - ((normal_gt * norm_est).sum(1, keepdim=True)[x["mask"]]).mean() )#* x["mask"]).sum() / (x["mask"] + 1e-7).sum()
            loss_distance = LOSS_DIST_WEIGHT * torch.abs(dist_gt- dist_est)[x["mask"]].mean()

            # Segmentation Loss
            #segment, planar_mask, dissimilarity_map = compute_seg(x["pixel_values"], norm_est, dist_est[:, 0])
            #loss_grad_normal, loss_grad_distance = get_smooth_ND(norm_est, dist_est, planar_mask)

            #loss_seg = 0.01 * (loss_grad_distance + loss_grad_normal)

            loss = loss_depth + loss_uncer + loss_normal + loss_distance #+ loss_seg
            loss = loss.mean()

            loss.backward()
            optimizer.step()
            
            #loop.set_postfix(loss=loss.item())
            custom_message = "Depth: {:.3g}, ".format(loss_depth.item())
            custom_message += "Uncer: {:.3g}, ".format(loss_uncer.item())
            custom_message += "Normal: {:.3g}, ".format(loss_normal.item())
            custom_message += "Dist: {:.3g}, ".format(loss_distance.item())
            #custom_message += "Seg: {:.3g}".format(loss_seg.item())
            loop.set_postfix(message=custom_message)
        
        # Reduce learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] *= LR_DECAY
        print(param_group['lr'])
        torch.save(model.module.state_dict(), 'model.pth')
        
        model.eval()
        torch.cuda.empty_cache()
        tot_metric = [0 for _ in range(METRIC_CNT)]
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
                    assert len(metric) == METRIC_CNT
                    for i in range(METRIC_CNT): tot_metric[i] += metric[i].cpu().detach().item()
                    cnt+=1

        for i in range(METRIC_CNT): tot_metric[i]/=cnt
        print(tot_metric)
        with open("metric.csv", mode='a', newline='') as file:  # Open in append mode
            writer = csv.writer(file)
            writer.writerow(tot_metric)  # Write the new row only
        
    dist.destroy_process_group()

def run_ddp(world_size):
    # We spawn the processes for each GPU using Python's multiprocessing
    mp.spawn(main, nprocs=world_size, args=(world_size,))

if __name__ == '__main__':
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    world_size = torch.cuda.device_count()
    run_ddp(world_size)