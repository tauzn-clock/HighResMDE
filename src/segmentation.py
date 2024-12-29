import torch
from skimage.segmentation import all_felzenszwalb as felz_seg

def normalize(a):
    return (a - a.min())/(a.max() - a.min() + 1e-8)

def compute_seg(rgb, aligned_norm, D):
        """
        inputs:
            rgb                 b, 3, H, W
            aligned_norm        b, 3, H, W
            D                   b, H, W

        outputs:
            segment                b, 1, H, W
            planar mask        b, 1, H, W
        """
        b, _, h, w  = rgb.shape
        device = rgb.device

        # compute cost
        #pdist = nn.PairwiseDistance(p=2, dim = 1)
        def pdist(x,y):
            return torch.linalg.norm(x-y, ord=2, dim=1)

        D_down = abs(D[:, 1:] - D[:, :-1])
        D_right = abs(D[:, :, 1:] - D[:, :, :-1])
        norm_down = pdist(aligned_norm[:, :, 1:], aligned_norm[:, :, :-1])
        norm_right = pdist(aligned_norm[:, :, :, 1:], aligned_norm[:, :, :, :-1])

        D_down = torch.stack([normalize(D_down[i]) for i in range(b)])
        norm_down = torch.stack([normalize(norm_down[i]) for i in range(b)])

        D_right = torch.stack([normalize(D_right[i]) for i in range(b)])
        norm_right = torch.stack([normalize(norm_right[i]) for i in range(b)])

        normD_down = D_down + norm_down
        normD_right = D_right + norm_right

        normD_down = torch.stack([normalize(normD_down[i]) for i in range(b)])
        normD_right = torch.stack([normalize(normD_right[i]) for i in range(b)])

        cost_down = normD_down
        cost_right = normD_right
        
        # get dissimilarity map visualization
        dst = cost_down[:,  :,  : -1] + cost_right[ :, :-1, :]
        
        # felz_seg
        cost_down_np = cost_down.detach().cpu().numpy()
        cost_right_np = cost_right.detach().cpu().numpy()
        segment = torch.stack([torch.from_numpy(felz_seg(normalize(cost_down_np[i]), normalize(cost_right_np[i]), 0, 0, h, w, scale=1, min_size=50)).to(device) for i in range(b)])
        segment += 1
        segment = segment.unsqueeze(1)
        
        # generate mask for segment with area larger than 200
        max_num = segment.max().item() + 1

        area = torch.zeros((b, max_num)).to(device)
        area.scatter_add_(1, segment.view(b, -1), torch.ones((b, 1, h, w)).view(b, -1).to(device))

        planar_area_thresh = 200
        valid_mask = (area > planar_area_thresh).float()
        planar_mask = torch.gather(valid_mask, 1, segment.view(b, -1))
        planar_mask = planar_mask.view(b, 1, h, w)

        planar_mask[:, :, :8, :] = 0
        planar_mask[:, :, -8:, :] = 0
        planar_mask[:, :, :, :8] = 0
        planar_mask[:, :, :, -8:] = 0

        return segment, planar_mask, dst.unsqueeze(1)


def get_smooth_ND(normal, distance, planar_mask):
    
    """Computes the smoothness loss for normal and distance
    """
    grad_normal_x = torch.mean(torch.abs(normal[:, :, :, :-1] - normal[:, :, :, 1:]), 1, keepdim=True)
    grad_normal_y = torch.mean(torch.abs(normal[:, :, :-1, :] - normal[:, :, 1:, :]), 1, keepdim=True)

    grad_distance_x = torch.abs(distance[:, :, :, :-1] - distance[:, :, :, 1:])
    grad_distance_y = torch.abs(distance[:, :, :-1, :] - distance[:, :, 1:, :])

    planar_mask_x = planar_mask[:, :, :, :-1]
    planar_mask_y = planar_mask[:, :, :-1, :]
    
    loss_grad_normal = (grad_normal_x * planar_mask_x).sum() / planar_mask_x.sum() + (grad_normal_y * planar_mask_y).sum() / planar_mask_y.sum()
    loss_grad_distance = (grad_distance_x * planar_mask_x).sum() / planar_mask_x.sum() + (grad_distance_y * planar_mask_y).sum() / planar_mask_y.sum()
    
    return loss_grad_normal, loss_grad_distance

def get_dist_laplace_kernel(dist):

    dist_pad = torch.nn.functional.pad(dist, (1, 1, 1, 1))
    dist_laplace = torch.abs(dist_pad[:, :, 2:, 1:-1] + dist_pad[:, :, :-2, 1:-1] + -2 * dist_pad[:,:,1:-1,1:-1])
    dist_laplace += torch.abs(dist_pad[:, :, 1:-1, 2:] + dist_pad[:, :, 1:-1, :-2] + -2 * dist_pad[:,:,1:-1,1:-1])

    return dist_laplace

def get_normal_laplace_kernel(normal):
    normal_pad = torch.nn.functional.pad(normal, (1, 1, 1, 1))
    normal_laplace = 1 - torch.sum((normal_pad[:,:,1:-1,1:-1] * normal_pad[:,:,2:,1:-1]),axis=1).unsqueeze(1)
    normal_laplace += 1 - torch.sum((normal_pad[:,:,1:-1,1:-1] * normal_pad[:,:,:-2,1:-1]),axis=1).unsqueeze(1)
    normal_laplace += 1 - torch.sum((normal_pad[:,:,1:-1,1:-1] * normal_pad[:,:,1:-1,2:]),axis=1).unsqueeze(1)
    normal_laplace += 1 - torch.sum((normal_pad[:,:,1:-1,1:-1] * normal_pad[:,:,1:-1,:-2]),axis=1).unsqueeze(1)
    return normal_laplace

def get_grad_loss(gt, pred):
    gt_pad = torch.nn.functional.pad(gt, (1, 1, 1, 1))
    pred_pad = torch.nn.functional.pad(pred, (1, 1, 1, 1))

    gt_grad_x = gt_pad[:, :, 2:, 1:-1] + gt_pad[:, :, :-2, 1:-1] + -2 * gt_pad[:,:,1:-1,1:-1]
    pred_grad_x = pred_pad[:, :, 2:, 1:-1] + pred_pad[:, :, :-2, 1:-1] + -2 * pred_pad[:,:,1:-1,1:-1]

    gt_grad_y = gt_pad[:, :, 1:-1, 2:] + gt_pad[:, :, 1:-1, :-2] + -2 * gt_pad[:,:,1:-1,1:-1]
    pred_grad_y = pred_pad[:, :, 1:-1, 2:] + pred_pad[:, :, 1:-1, :-2] + -2 * pred_pad[:,:,1:-1,1:-1]

    return torch.abs(pred_grad_x - gt_grad_x) + torch.abs(pred_grad_y - gt_grad_y)