from depth_to_pcd import depth_to_pcd
import numpy as np
import open3d as o3d
from PIL import Image
import csv
import os
from metrics import plane_ordering
from depth_to_pcd import depth_to_pcd
import matplotlib.pyplot as plt

def pcd_to_img(pcd):
    # Visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Add the point cloud to the visualizer
    vis.add_geometry(pcd)

    opt = vis.get_render_option()
    opt.point_size = 2

    view_control = vis.get_view_control()
    view_control.set_zoom(0.6) 
    view_control.rotate(100.0, 100.0)

    return np.array(vis.capture_screen_float_buffer(True))
def shrink_pcd_img(ori,gt,pred,INDEX):
    def get_bounding_box(mask):
        left = mask.shape[1]
        right = 0
        top = mask.shape[0]
        bottom = 0

        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if np.sum(mask[i,j]) < 3:
                    left = min(left, j)
                    right = max(right, j)
                    top = min(top, i)
                    bottom = max(bottom, i)
        
        return left, right, top, bottom

    left_ori, right_ori, top_ori, bottom_ori = get_bounding_box(ori)
    left_gt, right_gt, top_gt, bottom_gt = get_bounding_box(gt)
    left_pred, right_pred, top_pred, bottom_pred = get_bounding_box(pred)

    left = min(left_ori, left_gt, left_pred)
    right = max(right_ori, right_gt, right_pred)
    top = min(top_ori, top_gt, top_pred)
    bottom = max(bottom_ori, bottom_gt, bottom_pred)

    print(left, right, top, bottom)

    plt.imsave(f"paper_images/pcd_ori_{INDEX}.png", ori[top:bottom, left:right])
    plt.imsave(f"paper_images/pcd_gt_{INDEX}.png", gt[top:bottom, left:right])
    plt.imsave(f"paper_images/pcd_pred_{INDEX}.png", pred[top:bottom, left:right])

def fuse_coord_with_color(coord, color, mask):
    coord = coord.reshape(-1, 3)
    color = color.reshape(-1, 3)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coord[mask])
    pcd.colors = o3d.utility.Vector3dVector(color[mask])

    return pcd

def csv_to_depth(mask, csv, INTRINSICS):
    H, W = mask.shape
    mask = mask.flatten()

    x, y = np.meshgrid(np.arange(W), np.arange(H))
    x = x.flatten()
    y = y.flatten()
    fx, fy, cx, cy = INTRINSICS[0], INTRINSICS[5], INTRINSICS[2], INTRINSICS[6]
    Z = np.ones_like(x)
    x_3d = (x - cx) * Z / fx
    y_3d = (y - cy) * Z / fy
    POINTS = np.vstack((x_3d, y_3d, Z)).T
    DIRECTION_VECTOR = POINTS / (np.linalg.norm(POINTS, axis=1)[:, None]+1e-7)

    coord = np.zeros((H*W, 3))
    
    for i in range(1, mask.max()+1):
        norm = csv[i-1,:3]
        d = csv[i-1,3]
        plane_coord = (-d/(np.dot(DIRECTION_VECTOR, norm.T)+1e-7))[:,None]*DIRECTION_VECTOR
        coord[mask==i] = plane_coord[mask==i]

    return coord

def find_distance_for_gt_planes(coord, csv, mask):
    mask = mask.flatten()

    new_csv = []

    for i in range(1,mask.max()+1):
        norm = csv[i-1,:3]
        norm = norm / np.linalg.norm(norm)
        valid_mask = np.zeros_like(mask)
        valid_mask[mask==i] = 1
        valid_mask[coord[:,2] == 0] = 0
        dist = np.dot(coord[valid_mask==True], norm.T)
        new_csv.append([norm[0], norm[1], norm[2], -dist.mean()])

    return np.array(new_csv)

def transform(pcd):
    # Flip the point cloud
    tf = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    pcd.transform(tf)
    return pcd

if __name__ == '__main__':
    ROOT = "/scratchdata/nyu_plane"
    data_csv = "/HighResMDE/get_planes/ransac/config/nyu.csv"
    FOLDER = "new_gt_sigma_1"

    with open(data_csv, 'r') as f:
        reader = csv.reader(f)
        DATA = list(reader)

    INDEX = 1409

    data = DATA[INDEX]

    EPSILON = 1/float(data[6]) # Resolution
    R = float(data[7]) # Maximum Range

    INTRINSICS = np.array([float(data[2]), 0, float(data[4]), 0, 0, float(data[3]), float(data[5]), 0]) # fx, fy, cx, cy
    
    rgb = np.array(Image.open(os.path.join(ROOT, data[0]))) / 255.0
    depth = np.array(Image.open(os.path.join(ROOT, data[1])))/float(data[6])
    gt_mask = np.array(Image.open(f"{ROOT}/original_gt/{INDEX}.png"))
    pred_mask = np.array(Image.open(f"{ROOT}/{FOLDER}/{INDEX}.png"))
    with open(f"{ROOT}/original_gt/{INDEX}.csv", 'r') as f:
        reader = csv.reader(f)
        gt_csv = np.array(list(reader), dtype=np.float32)
    with open(f"{ROOT}/{FOLDER}/{INDEX}.csv", 'r') as f:
        reader = csv.reader(f)
        pred_csv = np.array(list(reader), dtype=np.float32)
    
    #Original RGB
    plt.imsave(f"paper_images/rgb_{INDEX}.png", rgb[45:471, 41:601])

    #GT Mask overlay
    fig, ax = plt.subplots()
    ax.imshow(rgb[45:471, 41:601])
    ax.imshow(gt_mask[45:471, 41:601], alpha=0.5, cmap='hsv')
    ax.axis('off')
    plt.savefig(f"paper_images/gt_mask_{INDEX}.png", bbox_inches='tight', pad_inches=0, transparent=True)

    #Pred Mask overlay
    fig, ax = plt.subplots()
    ax.imshow(rgb[45:471, 41:601])
    ax.imshow(pred_mask[45:471, 41:601], alpha=0.5, cmap='hsv')
    ax.axis('off')
    plt.savefig(f"paper_images/pred_mask_{INDEX}.png", bbox_inches='tight', pad_inches=0, transparent=True)
    
    #Original PCD
    coord, _ = depth_to_pcd(depth, INTRINSICS)
    pcd = fuse_coord_with_color(coord, rgb, coord[:,2] > 0)
    pcd_ori = transform(pcd)
    ori_pcd_img = pcd_to_img(pcd_ori)

    #GT PCD
    coord, _ = depth_to_pcd(depth, INTRINSICS)
    full_gt_csv = find_distance_for_gt_planes(coord, gt_csv, gt_mask)
    coord = csv_to_depth(gt_mask, full_gt_csv, INTRINSICS)
    pcd = fuse_coord_with_color(coord, rgb, coord[:,2] > 0)
    pcd_gt = transform(pcd)
    gt_pcd_img = pcd_to_img(pcd_gt)

    #Pred PCD
    coord = csv_to_depth(pred_mask, pred_csv, INTRINSICS)
    pcd = fuse_coord_with_color(coord, rgb, coord[:,2] > 0)
    pcd_pred = transform(pcd)
    pred_pcd_img = pcd_to_img(pcd_pred)

    shrink_pcd_img(ori_pcd_img,gt_pcd_img,pred_pcd_img,INDEX)

"""
Smallest RI:  1409
Largest VOI:  1207
Smallest SC:  1207
"""