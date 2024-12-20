import sys
sys.path.append('/HighResMDE/src')

import torch
from torch.utils.data import DataLoader
import tqdm
from torchvision import transforms

from model import Model, ModelConfig
from dataloader.BaseDataloader import BaseImageDataset
from dataloader.NYUDataloader import NYUImageData
from loss import silog_loss, get_metrics
from global_parser import global_parser
from infer_image import infer_image

import matplotlib.pyplot as plt

def eval(model, args, local_rank, test_dataloader):
    model.eval()
    torch.cuda.empty_cache()
    tot_metric = [0 for _ in range(args.metric_cnt)]
    cnt = 0
    with torch.no_grad():
        for _, x in enumerate(tqdm.tqdm(test_dataloader)):
            for k in x.keys():
                x[k] = x[k].to(local_rank)
                
            depth = infer_image(model, x)
            depth_gt = x["depth_values"]
            
            for b in range(x["max_depth"].shape[0]):
                metric = get_metrics(depth_gt[b], depth[b], x["mask"][b])
                assert len(metric) == args.metric_cnt
                for i in range(args.metric_cnt): tot_metric[i] += metric[i].cpu().detach().item()
                cnt+=1

    for i in range(args.metric_cnt): tot_metric[i]/=cnt
    print(tot_metric)

    print("Save test image")
    plt.imsave("best_depth_est.png", depth[0].cpu().detach().squeeze())

    return tot_metric

if __name__ == "__main__":
    args = global_parser()
    local_rank = "cuda"

    test_dataset = BaseImageDataset('test', NYUImageData, args.test_dir, args.test_csv)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, pin_memory=True)

    config = ModelConfig(args.model_size)
    model = Model(config).to(local_rank)

    print("Using ", args.pretrained_model)
    model.load_state_dict(torch.load(args.pretrained_model, weights_only=False))
    torch.cuda.empty_cache()
    #model.backbone.backbone.from_pretrained(model.config.swinv2_pretrained_path)
    # Freeze the encoder layers only
    #for param in model.backbone.backbone.parameters():  # 'backbone' is typically where the encoder layers reside
    #    param.requires_grad = False
    #torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

    total_metric = eval(model, args, local_rank, test_dataloader)
    print(total_metric)