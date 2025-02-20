import numpy as np

metric =  np.load("default_gt.npy")
#metric = np.load("noise_model.npy")
#metric = np.load("sigma_1.npy")

print(np.mean(metric, axis=0))

# Find percentage of images
print((metric[:,:,1]>0).sum(axis=0)/metric.shape[0])

# Find mean within the images that have at least one plane
for i in range(5):   
    
    found_semantic = 0
    found_semantic_with_plane = 0
    total_ratio = 0

    for j in range(metric.shape[0]):
        if metric[j,i].sum() != 0:
            found_semantic += 1
            if metric[j,i].sum() != 1:
                found_semantic_with_plane += 1
                total_ratio += metric[j,i,2]
    print(found_semantic_with_plane, found_semantic, total_ratio/found_semantic_with_plane)


