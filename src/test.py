import sys
sys.path.append('/HighResMDE/src')

from PIL import Image
import torch
import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F

import matplotlib.pyplot as plt
from model import Model, ModelConfig
from dataloader.BaseDataloader import BaseImageDataset
from dataloader.NYUDataloader import NYUImageData

#torch.manual_seed(42)
MODEL_PATH = "./model.pth"
BATCH_SIZE = 4
MODEL_SIZE = "large07"
SWINV2_SPECIFIC_PATH = None #"microsoft/swinv2-tiny-patch4-window8-256"

device = "cuda:0"

test_dataset = BaseImageDataset('test', NYUImageData, '/scratchdata/nyu_data/', '/HighResMDE/src/nyu_train.csv')
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, pin_memory=True)

for i, data in enumerate(test_dataloader):
    break
plt.imshow(data["pixel_values"][0].squeeze().permute(1, 2, 0))

config =  ModelConfig(MODEL_SIZE)
config.batch_size = BATCH_SIZE
config.height = 480//4
config.width = 640//4
if not SWINV2_SPECIFIC_PATH is None: config.swinv2_pretrained_path = SWINV2_SPECIFIC_PATH
model = Model(config).to(device)
model.backbone.backbone.from_pretrained(model.config.swinv2_pretrained_path)
# Freeze the encoder layers only
for param in model.backbone.backbone.parameters():  # 'backbone' is typically where the encoder layers reside
    param.requires_grad = False
#torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))

for k in data.keys(): data[k] = data[k].to(device)
d1_list, u1, d2_list, u2, norm_est, dist_est = model(data)

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