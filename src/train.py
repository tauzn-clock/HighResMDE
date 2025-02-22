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
from torchvision.transforms import GaussianBlur

from model import Model, ModelConfig
from dataloader.BaseDataloader import BaseImageDataset
from dataloader.NYUDataloader import NYUImageData
from layers.DN_to_distance import DN_to_distance
from layers.depth_to_normal import Depth2Normal
from loss import silog_loss, get_metrics
from segmentation import compute_seg, get_smooth_ND, get_dist_laplace_kernel, get_normal_laplace_kernel, get_grad_loss
from global_parser import global_parser
from eval_metric import eval
from plane_estimation import normal_to_planes

import matplotlib.pyplot as plt

from CutMix import CutMix, CutFlip


torch.manual_seed(42)

def init_process_group(local_rank, world_size):
    dist.init_process_group(
        backend='nccl',  # Use NCCL backend for multi-GPU communication
        rank=local_rank,
        world_size=world_size
    )
    torch.cuda.set_device(local_rank)  # Set the GPU device for the current process

def main(local_rank, world_size):
    args = global_parser()
    if args.cutmix:
        print("Using CutMix with prob ", args.cut_prob)
    if args.cutflip:
        print("Using CutFlip with prob ", args.cut_prob)
    init_process_group(local_rank, world_size)

    train_dataset = BaseImageDataset('train', NYUImageData, args.train_dir, args.train_csv)
    train_sampler = torch.utils.data.DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, sampler=train_sampler)

    test_dataset = BaseImageDataset('test', NYUImageData, args.test_dir, args.test_csv)
    test_sampler = torch.utils.data.DistributedSampler(test_dataset, num_replicas=world_size, rank=local_rank)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, pin_memory=True, sampler=test_sampler)

    csv_file = [["silog", "abs_rel", "log10", "rms", "sq_rel", "log_rms", "d1", "d2", "d3"]]
    with open(args.metric_save_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(csv_file)

    config = ModelConfig(args.model_size)
    model = Model(config).to(local_rank)
    if not args.pretrained_model is None: 
        print("Using ", args.pretrained_model)
        model.load_state_dict(torch.load(args.pretrained_model, weights_only=False))
        torch.cuda.empty_cache()
    # Freeze the encoder layers only
    if not args.encoder_grad:
        for param in model.backbone.parameters():  # 'backbone' is typically where the encoder layers reside
            param.requires_grad = False
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
   
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    silog_criterion = silog_loss(variance_focus=args.var_focus).to(local_rank)
    dn_to_distance = DN_to_distance(args.batch_size, args.height, args.width).to(local_rank)
    normal_estimation = Depth2Normal(local_rank)
    gaussian_blur = GaussianBlur(kernel_size=args.gaussian_blur_kernel, sigma=(args.gaussian_blur_kernel-1)/6)

    for epoch in range(args.total_epoch):
        model.train()

        loop = tqdm.tqdm(train_dataloader, desc=f"Epoch {epoch+1}", unit="batch")
        for itr, x in enumerate(loop):
            optimizer.zero_grad()
            with torch.no_grad():
                for k in x.keys():
                    x[k] = x[k].to(local_rank)
                if x["depth_values"].shape[0] != args.batch_size: continue # Hacky solution to deal with batch size issue
                
                # Estimate GT normal and distance

                depth_gt = x["depth_values"] #Unit: m
                normal_gt, x["mask"] = normal_estimation(depth_gt, x["camera_intrinsics"], x["mask"], 1.0) # Intrinsic needs to be in mm, ideally change depth_gt to mm for consistency, skip for speed
                for b in range(len(normal_gt)):
                    normal_gt[b] = gaussian_blur(normal_gt[b])
                normal_gt = normal_gt * x["mask"]
                normal_gt = F.normalize(normal_gt, dim=1, p=2) #Unit: none, normalised
                dist_gt = dn_to_distance(depth_gt, normal_gt, x["camera_intrinsics_inverted"]) #Camera intrinsic needs to be in mm, but dist_gt is in m, probably dont need to scale depth_gt but just to be safe

            # CutMix
            if args.cutmix:
                if torch.rand(1) < args.cut_prob:
                    x["pixel_values"], x["depth_values"], x["mask"], normal_gt, dist_gt = CutMix(x["pixel_values"], x["depth_values"], x["mask"], normal_gt, dist_gt)

            # CutFlip
            if args.cutflip:
                if torch.rand(1) < args.cut_prob:
                    x["pixel_values"], x["depth_values"], x["mask"], normal_gt, dist_gt = CutFlip(x["pixel_values"], x["depth_values"], x["mask"], normal_gt, dist_gt)

            # Forward pass
            d1_list, u1, d2_list, u2, norm_est, dist_est = model(x)

            # Depth Loss

            loss_depth1_0 = silog_criterion(d1_list[0], depth_gt, x["mask"])
            loss_depth2_0 = silog_criterion(d2_list[0], depth_gt, x["mask"])

            loss_depth1 = 0
            loss_depth2 = 0
            weights_sum = 0
            for i in range(len(d1_list) - 1):
                loss_depth1 += (args.var_focus**(len(d1_list)-i-2)) * silog_criterion(d1_list[i + 1], depth_gt, x["mask"])
                loss_depth2 += (args.var_focus**(len(d2_list)-i-2)) * silog_criterion(d2_list[i + 1], depth_gt, x["mask"])
                weights_sum += args.var_focus**(len(d1_list)-i-2)
            
            if (epoch < args.initial_epoch):
                loss_depth =  args.loss_depth_weight * (loss_depth1_0 + loss_depth2_0)
            else:
                loss_depth =  args.loss_depth_weight * ((loss_depth1 + loss_depth2) / weights_sum + loss_depth1_0 + loss_depth2_0 )
            
            # Uncertainty Loss

            uncer1_gt = torch.exp(-5 * torch.abs(depth_gt - d1_list[0].detach()) / (depth_gt + d1_list[0].detach() + 1e-7))
            uncer2_gt = torch.exp(-5 * torch.abs(depth_gt - d2_list[0].detach()) / (depth_gt + d2_list[0].detach() + 1e-7))
            
            loss_uncer1 = torch.abs(u1-uncer1_gt)[x["mask"]].mean()
            loss_uncer2 = torch.abs(u2-uncer2_gt)[x["mask"]].mean()

            loss_uncer =  args.loss_uncer_weight * (loss_uncer1 + loss_uncer2)

            loss_normal = args.loss_normal_weight * (1 - ((normal_gt * norm_est).sum(1, keepdim=True)[x["mask"]]).mean() )#* x["mask"]).sum() / (x["mask"] + 1e-7).sum()
            loss_distance = args.loss_dist_weight * torch.abs(dist_gt- dist_est)[x["mask"]].mean()

            # Segmentation Loss
            #segment, planar_mask, dissimilarity_map = compute_seg(x["pixel_values"], norm_est, dist_est[:, 0])
            #loss_grad_normal, loss_grad_distance = get_smooth_ND(norm_est, dist_est, planar_mask)
            
            dist_grad = get_dist_laplace_kernel(dist_est)
            norm_grad = get_normal_laplace_kernel(norm_est)

            loss_seg_dist = 0
            loss_seg_norm = 0

            PLANE_CNT = 32
            K_MEAN_ITERATION = 20

            x["plane_values"] = normal_to_planes(norm_est, dist_est, x["mask"], PLANE_CNT, K_MEAN_ITERATION)

            for b in range(args.batch_size):
                for i in range(1, PLANE_CNT+1):
                    tmp_a = dist_grad[b][x["plane_values"][b]==i].mean() 
                    tmp_b = norm_grad[b][x["plane_values"][b]==i].mean()
                    if tmp_a > 0:
                        loss_seg_dist += tmp_a
                    if tmp_b > 0:
                        loss_seg_norm += tmp_b

            loss_seg_dist = args.loss_seg_dist_weight * loss_seg_dist
            loss_seg_norm = args.loss_seg_norm_weight * loss_seg_norm            

            #loss_seg = 0.01 * (loss_grad_distance + loss_grad_normal)

            # Grad Loss
            loss_grad = 100 * get_grad_loss(depth_gt, (d1_list[0]+d2_list[0])/2)[x["mask"]].mean()

            loss = loss_depth + loss_uncer + loss_normal + loss_distance + loss_seg_dist + loss_seg_norm #+ loss_grad
            loss = loss.mean()

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10, norm_type=2)
            optimizer.step()
            
            #loop.set_postfix(loss=loss.item())
            custom_message = "Depth: {:.3g}, ".format(loss_depth.item())
            custom_message += "Uncer: {:.3g}, ".format(loss_uncer.item())
            custom_message += "Normal: {:.3g}, ".format(loss_normal.item())
            custom_message += "Dist: {:.3g}, ".format(loss_distance.item())
            custom_message += "Seg D: {:.3g}, ".format(loss_seg_dist.item())
            custom_message += "Seg N: {:.3g}".format(loss_seg_norm.item())
            custom_message += "Grad: {:.3g}".format(loss_grad.item())
            loop.set_postfix(message=custom_message)
        
        # Reduce learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] *= args.lr_decay
        print(param_group['lr'])
        torch.save(model.module.state_dict(), args.model_save_path)
        
        tot_metric = eval(model, args, local_rank, test_dataloader)
        with open(args.metric_save_path, mode='a', newline='') as file:  # Open in append mode
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