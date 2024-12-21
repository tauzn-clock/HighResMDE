import numpy as np
import torch

def CutMix(img, depth, mask, normal, dist):
    B, _, H, W = img.shape

    top = np.random.randint(int(0.20 * H), int(0.40 * H))
    left = np.random.randint(int(0.20 * W), int(0.40 * W))

    bottom = np.random.randint(int(0.6 * H), int(0.8 * H))
    right = np.random.randint(int(0.6 * W), int(0.8 * W))

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

def CutFlip(img, depth, mask, normal, dist):
    B, _, H, W = img.shape

    height = np.random.randint(int(0.2*H), int(0.8*H))

    img = torch.cat((img[:,:,height:,:], img[:,:,:height,:]), dim=2)
    depth = torch.cat((depth[:,:,height:,:], depth[:,:,:height,:]), dim=2)
    mask = torch.cat((mask[:,:,height:,:], mask[:,:,:height,:]), dim=2)
    normal = torch.cat((normal[:,:,height:,:], normal[:,:,:height,:]), dim=2)

    return img, depth, mask, normal, dist
