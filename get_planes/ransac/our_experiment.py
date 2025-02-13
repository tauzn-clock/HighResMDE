import numpy as np
from information_estimation import plane_ransac
from visualise import visualise_mask
from synthetic_test import set_depth, open3d_find_planes
import matplotlib.pyplot as plt
import open3d as o3d
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

def save_mask(mask, filepath):
    def hsv_to_rgb(h, s, v):
        """
        Convert HSV to RGB.

        :param h: Hue (0 to 360)
        :param s: Saturation (0 to 1)
        :param v: Value (0 to 1)
        :return: A tuple (r, g, b) representing the RGB color.
        """
        h = h / 360  # Normalize hue to [0, 1]
        c = v * s  # Chroma
        x = c * (1 - abs((h * 6) % 2 - 1))  # Temporary value
        m = v - c  # Match value

        if 0 <= h < 1/6:
            r, g, b = c, x, 0
        elif 1/6 <= h < 2/6:
            r, g, b = x, c, 0
        elif 2/6 <= h < 3/6:
            r, g, b = 0, c, x
        elif 3/6 <= h < 4/6:
            r, g, b = 0, x, c
        elif 4/6 <= h < 5/6:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x

        # Adjust to match value
        r = (r + m) 
        g = (g + m) 
        b = (b + m)

        return r, g, b

    H, W = mask.shape
    color = np.zeros((H, W, 3))
    for i in range(1, mask.max()+1):
        color[mask==i] = hsv_to_rgb(i/mask.max()*360, 1, 1)
    
    plt.imsave(filepath, color)

NOISE_LEVEL = 2
SIGMA_LEVEL = 0.002

ROOT = "/HighResMDE/get_planes/4_planes"
H = 480
W = 640
R = 10
EPSILON = 0.001
SIGMA = np.ones(H*W) * SIGMA_LEVEL
MAX_PLANE = 10
CONFIDENCE = 0.99
INLIER_THRESHOLD = 0.25

INTRINSICS = np.array([500, 0, 320, 0, 0, 500, 240])
N = 4
depth = np.zeros((H,W))
mask = np.zeros((H,W),dtype=int)
for i in range(N):
    plane_mask = np.zeros((H,W),dtype=bool)
    plane_mask[:,i*W//N:(i+1)*W//N] = True
    depth += set_depth(np.ones((H,W)),INTRINSICS, plane_mask, [0,0,1], -0.1 * i - 2.5)
    mask[plane_mask] = i+1

# Random assignment
random_mask = np.random.randint(100, size=(H,W))
depth[random_mask>30] = np.random.randint(0.24*R//EPSILON, 0.29*R//EPSILON, size=(H,W))[random_mask>30] * EPSILON
mask[random_mask>30] = 0

plt.imsave(f"{ROOT}/{NOISE_LEVEL}_depth.png", depth, cmap='gray')
save_mask(mask, f"{ROOT}/{NOISE_LEVEL}_gt.png")

visualise_mask(depth, mask, INTRINSICS)

mask, planes = open3d_find_planes(depth, INTRINSICS, SIGMA_LEVEL, CONFIDENCE, INLIER_THRESHOLD, MAX_PLANE, verbose=True)
print(mask)
save_mask(mask, f"{ROOT}/{NOISE_LEVEL}_{SIGMA_LEVEL}_default.png")

print(mask.max())
print(planes)

visualise_mask(depth, mask, INTRINSICS)

#R = 10 * 0.05
information, mask, planes = plane_ransac(depth, INTRINSICS, R, EPSILON, SIGMA, CONFIDENCE, INLIER_THRESHOLD, MAX_PLANE, verbose=True)

print(information)
print(planes)

smallest = np.argmin(information)

mask[mask>smallest] = 0
print(mask.max())
#save_mask(mask, f"{ROOT}/{NOISE_LEVEL}_{SIGMA_LEVEL}_ours.png")

visualise_mask(depth, mask, INTRINSICS)

