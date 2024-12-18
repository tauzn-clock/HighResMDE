from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

IMG_PATH = "/scratchdata/nyu_depth_v2/sync/bedroom_0130/rgb_00000.jpg"
DEPTH_PATH = "/scratchdata/nyu_depth_v2/sync/bedroom_0130/sync_depth_00000.png"

img = Image.open(IMG_PATH)
depth = Image.open(DEPTH_PATH)

img = np.array(img)
depth = np.array(depth)

print(depth.max())

plt.imsave("rgb.png", img)
plt.imsave("depth.png", depth)