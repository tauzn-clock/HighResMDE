import scipy.io
import mat73
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

test = cv2.imread("/scratchdata/nyu_plane/labels/0.png", cv2.IMREAD_GRAYSCALE)
plt.imsave("test.png",test)

for i in range(1449):
    mat = scipy.io.loadmat(f'/scratchdata/nyud/segmentation/img_{5001+i}.mat')
    print(mat["segmentation"].shape)
    print(mat["segmentation"].max(), mat["segmentation"].min())

    label = np.zeros((480, 640), dtype=np.uint8)
    label[46:471, 41:601] = mat["segmentation"]

    cv2.imwrite(f'/scratchdata/nyu_plane/labels/{i}.png', label)