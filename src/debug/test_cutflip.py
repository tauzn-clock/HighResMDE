import numpy as np
import random
import copy
from PIL import Image
import matplotlib.pyplot as plt

def Cut_Flip(image, depth, normal):

    p = random.random()
    image_copy = copy.deepcopy(image)
    depth_copy = copy.deepcopy(depth)
    normal_copy = copy.deepcopy(normal)
    h, w, c = image.shape
    N = 2     # split numbers
    h_list = []
    h_interval_list = []   # hight interval
    for i in range(N-1):
        h_list.append(random.randint(int(0.2*h), int(0.8*h)))
    h_list.append(h)
    h_list.append(0)  
    h_list.sort()
    h_list_inv = np.array([h]*(N+1))-np.array(h_list)
    for i in range(len(h_list)-1):
        h_interval_list.append(h_list[i+1]-h_list[i])
    for i in range(N):
        image[h_list[i]:h_list[i+1], :, :] = image_copy[h_list_inv[i]-h_interval_list[i]:h_list_inv[i], :, :]
        depth[h_list[i]:h_list[i+1], :, :] = depth_copy[h_list_inv[i]-h_interval_list[i]:h_list_inv[i], :, :]
        normal[h_list[i]:h_list[i+1], :, :] = normal_copy[h_list_inv[i]-h_interval_list[i]:h_list_inv[i], :, :]

    return image, depth, normal

image = Image.open("/scratchdata/nyu_depth_v2/official_splits/test/bathroom/rgb_00045.jpg")
image = np.array(image)

test, _, _ = Cut_Flip(image, image, image)
plt.imsave("test.png", test)
