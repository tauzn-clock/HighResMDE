from PIL import Image
import numpy as np

PATH = "/scratchdata/nyu_depth_v2/sync/planes_8/bedroom_0130/sync_depth_00000.png"

depth_image = Image.open(PATH)

depth_image = np.array(depth_image)


print(depth_image.shape)
print(depth_image.max())