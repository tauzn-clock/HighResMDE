import numpy as np
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

    #final_mask[depth == 0] = -1

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

    #final_mask[final_mask == -1] = 0

    return final_mask, np.array(final_planes)
