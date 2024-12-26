import torch

def get_dist_laplace_kernel(dist):

    dist_pad = torch.nn.functional.pad(dist, (1, 1, 1, 1))
    dist_laplace = -4 * dist_pad[:,:,1:-1,1:-1]
    dist_laplace += dist_pad[:, :, 2:, 1:-1]
    dist_laplace += dist_pad[:, :, :-2, 1:-1]
    dist_laplace += dist_pad[:, :, 1:-1, 2:]
    dist_laplace += dist_pad[:, :, 1:-1, :-2]
    dist_laplace = torch.abs(dist_laplace)

    return dist_laplace

def get_normal_laplace_kernel(normal):
    normal_pad = torch.nn.functional.pad(normal, (1, 1, 1, 1))
    normal_laplace = 1 - torch.sum((normal_pad[:,:,1:-1,1:-1] * normal_pad[:,:,2:,1:-1]),axis=1).unsqueeze(1)
    normal_laplace += 1 - torch.sum((normal_pad[:,:,1:-1,1:-1] * normal_pad[:,:,:-2,1:-1]),axis=1).unsqueeze(1)
    normal_laplace += 1 - torch.sum((normal_pad[:,:,1:-1,1:-1] * normal_pad[:,:,1:-1,2:]),axis=1).unsqueeze(1)
    normal_laplace += 1 - torch.sum((normal_pad[:,:,1:-1,1:-1] * normal_pad[:,:,1:-1,:-2]),axis=1).unsqueeze(1)
    return normal_laplace


test = torch.Tensor([[1,0,0],[0,1,0],[0,0,1],[0,0,-1],[0.707,0.707,0],[0,0,1]])
test = torch.transpose(test,0,1)
test = test.reshape((3,3,2))
test = torch.stack([test])

print(test.shape)
print(test[0,:,0,0])
print(test[0,:,0,1])
print(test[0,:,1,0])
print(test[0,:,1,1])
print(test[0,:,2,0])
print(test[0,:,2,1])

tmp = get_normal_laplace_kernel(test)
print(tmp)