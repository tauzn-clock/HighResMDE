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
MODEL_PATH = "./large_model.pth"
BATCH_SIZE = 4

device = "cuda:0"

test_dataset = BaseImageDataset('test', NYUImageData, '/scratchdata/nyu_data/', '/HighResMDE/src/nyu_test.csv')
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, pin_memory=True)

for i, data in enumerate(test_dataloader):
    break
plt.imshow(data["pixel_values"][0].squeeze().permute(1, 2, 0))

config =  ModelConfig("large07")
config.batch_size = BATCH_SIZE
config.height = 480//4
config.width = 640//4
model = Model(config).to(device)
model.backbone.backbone.from_pretrained("microsoft/swinv2-large-patch4-window12-192-22k")
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))

for k in data.keys(): data[k] = data[k].to(device)
d1_list, u1, d2_list, u2, norm_est, dist_est = model(data)

for i in range(len(d1_list)):
    plt.imshow((d1_list[i][0]+d2_list[i][0]).cpu().detach().squeeze())
    plt.savefig(f"output_image_{i}.png", bbox_inches='tight', dpi=300)

print((d1_list[-1][0]+d2_list[-1][0]))