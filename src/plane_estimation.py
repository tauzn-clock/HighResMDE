import torch
import torch.nn.functional as F

def normal_to_planes(normal, dist, mask, PLANE_CNT=128, K_MEAN_ITERATION=100):
    assert normal.device == mask.device

    device = normal.device

    def recluster(normal, dist, store_normal, store_dist, PLANE_CNT):
        normal_flatten = normal.view(3, -1)
        normal_distance_function = 1 - torch.matmul(store_normal.t(), normal_flatten)
        normal_distance_function = normal_distance_function.view(PLANE_CNT, *normal.shape[1:])

        dist_flatten = dist.view(-1)
        dist_distance_function = torch.abs(store_dist.t() - dist_flatten)
        dist_distance_function = dist_distance_function.view(PLANE_CNT, *normal.shape[1:])

        distance_function = normal_distance_function + dist_distance_function

        new_plane = torch.argmin(distance_function, dim=0) + 1

        return new_plane

    B, _, H, W = normal.shape

    with torch.no_grad():

        # Generate Index Mesh
        coords_h = torch.arange(H).to(device)
        coords_w = torch.arange(W).to(device)
        index_mesh = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))

        plane = torch.randint(1, PLANE_CNT+1, (B,1,H,W)).to(device)
        plane = plane * mask

        # Randomly Select Points
        for b in range(B):
            index_valid = index_mesh[:, mask[b].squeeze()]
            index_select = index_valid[:, :PLANE_CNT]

            store_normal = normal[b,:,index_select[0,:],index_select[1,:]]
            store_dist = dist[b,:,index_select[0,:],index_select[1,:]]

            plane[b] = recluster(normal[b], dist[b], store_normal, store_dist, PLANE_CNT) * mask[b]

        for b in range(B):
            for em_itr in range(K_MEAN_ITERATION):
        
                # Average
                for i in range(1, PLANE_CNT+1):
                    index_mask = (plane[b]==i).float()
                    if index_mask.sum() == 0: continue
                    normal_mean = normal[b] * index_mask.unsqueeze(0)
                    normal_mean = normal_mean.squeeze(0)
                    normal_mean = normal_mean.sum(dim=(1,2)) / index_mask.sum()
                    dist_mean = dist[b] * index_mask.unsqueeze(0)
                    dist_mean = dist_mean.sum() / index_mask.sum()
                    store_normal[:, i-1] = normal_mean
                    store_dist[:,i-1] = dist_mean

                store_normal = F.normalize(store_normal, dim=0)
                
                plane[b] = recluster(normal[b], dist[b], store_normal, store_dist, PLANE_CNT) * mask[b]
            
        return plane