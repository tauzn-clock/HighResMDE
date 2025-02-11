import numpy as np
from information_estimation import plane_ransac
from visualise import visualise_mask
from depth_to_pcd import depth_to_pcd
import open3d as o3d

def set_depth(depth,intrinsic,mask,normal,distance):
    normal = np.array(normal)
    normal = normal / np.linalg.norm(normal)

    H,W = depth.shape
    Z = depth.flatten()

    x, y = np.meshgrid(np.arange(W), np.arange(H))
    x = x.flatten()
    y = y.flatten()
    fx, fy, cx, cy = intrinsic[0], intrinsic[5], intrinsic[2], intrinsic[6]
    x_3d = (x - cx) * Z / fx
    y_3d = (y - cy) * Z / fy
    POINTS = np.vstack((x_3d, y_3d, Z)).T

    DIRECTION_VECTOR = POINTS / (np.linalg.norm(POINTS, axis=1)[:, None]+1e-7)

    distance = (-distance/np.dot(DIRECTION_VECTOR, normal.T)) * DIRECTION_VECTOR[:,2]
    
    distance = distance.reshape(H,W)

    return distance*mask

def open3d_find_planes(depth, INTRINSICS, SIGMA, CONFIDENCE, INLIER_THRESHOLD, MAX_PLANE, verbose=True):
    final_mask = np.zeros_like(depth,dtype=int)
    final_planes = []

    points, index = depth_to_pcd(depth, INTRINSICS)

    pcd = o3d.geometry.PointCloud()
    
    ITERATION = int(np.log(1 - CONFIDENCE) / np.log(1 - INLIER_THRESHOLD**3))
    print("Iteration: ", ITERATION)

    for i in range(MAX_PLANE):
        pcd.points = o3d.utility.Vector3dVector(points[final_mask.flatten()==0])
        plane_model, plane_inliers = pcd.segment_plane(SIGMA, 3, ITERATION)
        [a, b, c, d] = plane_model
        if verbose:
            print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

        plane_index = index[final_mask.flatten()==0][plane_inliers]
        final_mask[plane_index[:,0],plane_index[:,1]] = i+1
        
        final_planes.append([a,b,c,d])

    return final_mask, np.array(final_planes)

H = 480
W = 640
R = 10
EPSILON = 0.001
SIGMA = np.ones(H*W) * 0.002
MAX_PLANE = 8
CONFIDENCE = 0.99
INLIER_THRESHOLD = 0.15

INTRINSICS = np.array([500, 0, 320, 0, 0, 500, 240])
N = 4
depth = np.zeros((H,W))
for i in range(N):
    mask = np.zeros((H,W),dtype=bool)
    mask[:,i*W//N:(i+1)*W//N] = True
    depth += set_depth(np.ones((H,W)),INTRINSICS, mask, [0,0,1], 2 * i - 10)
depth += np.random.randint(10, size=(H,W)) * EPSILON

# Random assignment
random_mask = np.random.randint(2, size=(H,W))
depth[random_mask==0] = np.random.randint(R//EPSILON, size=(H,W))[random_mask==0] * EPSILON

mask, planes = open3d_find_planes(depth, INTRINSICS, 0.02, CONFIDENCE, INLIER_THRESHOLD, 8, verbose=True)

print(mask.max())
print(planes)

visualise_mask(depth, mask, INTRINSICS)

information, mask, planes = plane_ransac(depth, INTRINSICS, R, EPSILON, SIGMA, CONFIDENCE, INLIER_THRESHOLD, MAX_PLANE, verbose=True)

print(information)
print(planes)

smallest = np.argmin(information)

mask[mask>smallest] = 0
print(mask.max())

visualise_mask(depth, mask, INTRINSICS)

