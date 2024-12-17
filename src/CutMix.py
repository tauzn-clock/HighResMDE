import numpy as np
import torch

def CutMix(img, depth, mask, normal, dist):
    B, _, H, W = img.shape

    top = np.random.randint(int(0.1 * H), int(0.3 * H))
    left = np.random.randint(int(0.1 * W), int(0.3 * W))

    bottom = np.random.randint(int(0.7 * H), int(0.9 * H))
    right = np.random.randint(int(0.7 * W), int(0.9 * W))

    img_a = img.clone()
    depth_a = depth.clone()
    mask_a = mask.clone()
    normal_a = normal.clone()
    dist_a = dist.clone()

    for b in range(B):
        img_a[b, :, top:bottom, left:right] = img[(b+1)%B, :, top:bottom, left:right]
        depth_a[b, :, top:bottom, left:right] = depth[(b+1)%B, :, top:bottom, left:right]
        mask_a[b, :, top:bottom, left:right] = mask[(b+1)%B, :, top:bottom, left:right]
        normal_a[b, :, top:bottom, left:right] = normal[(b+1)%B, :, top:bottom, left:right]
        dist_a[b, :, top:bottom, left:right] = dist[(b+1)%B, :, top:bottom, left:right]
        
    return img_a, depth_a, mask_a, normal_a, dist_a
