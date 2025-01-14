import torch
import torch.nn.functional as F

def normal_to_planes(normal, dist, mask, PLANE_CNT=128, K_MEAN_ITERATION=100, PRUNE_POST_CLUSTERING=True, PRUNE_PIXEL_RATIO = 0.1, PRUNE_DIST_THRESHOLD = 0.1):
    assert normal.device == mask.device

    device = normal.device

    def nd_distance(normal_a, normal_b, dist_a, dist_b):
        normal_dist = 1 - torch.matmul(normal_a.t(), normal_b)
        dist_dist = torch.abs(dist_a.t() - dist_b)
        return normal_dist + dist_dist


    def recluster(normal, dist, store_normal, store_dist, PLANE_CNT):
        normal_flatten = normal.view(3, -1)
        dist_flatten = dist.view(1, -1)

        distance_function = nd_distance(normal_flatten, store_normal, dist_flatten, store_dist)
        distance_function = distance_function.t()
        H, W = normal.shape[-2:]

        distance_function = distance_function.reshape(PLANE_CNT, H, W)

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

            if PRUNE_POST_CLUSTERING:
                # Prunning Loop
                pixel_cnt = torch.zeros(PLANE_CNT)  
                for value in range(1, PLANE_CNT+1):
                    pixel_cnt[value - 1] = (plane[b] == value).sum()
                pixel_cnt[pixel_cnt==0] = float("inf")
                while True:
                    min_index= torch.argmin(pixel_cnt)   
                    min_count = pixel_cnt[min_index]
                    if (min_count > PRUNE_PIXEL_RATIO * H * W):
                        break
                    distance_function = nd_distance(store_normal, store_normal[:,min_index].view(3,-1), store_dist, store_dist[:,min_index].view(1,-1))
                    distance_function = distance_function.t().squeeze()
                    distance_function[pixel_cnt==float("inf")] = float("inf")
                    distance_function[min_index] = float("inf")
                    while True:
                        closest_plane = torch.argmin(distance_function, dim=0)
                        if distance_function[closest_plane] == float("inf"):
                            break
                        if distance_function[closest_plane] < PRUNE_DIST_THRESHOLD:
                            pixel_cnt[closest_plane] += min_count
                            plane[b][plane[b] == min_index+1] = closest_plane + 1
                            break
                            
                        else:
                            distance_function[closest_plane] = float("inf")
                        
                    pixel_cnt[min_index] = float("inf")            

                # Reindex in plane
                pt = 1
                for i in range(1, PLANE_CNT+1):
                    if (plane[b] == i).sum() == 0:
                        continue
                    plane[b][plane[b] == i] = pt
                    pt += 1

        return plane