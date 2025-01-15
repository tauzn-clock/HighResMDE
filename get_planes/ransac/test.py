import numpy as np
import open3d as o3d
import random

EPSILON = 0.1 # Resolution
R = 10 # Maximum Range
SIGMA = 1 # Normal std

# Parameters for the flat plane
num_points = 1000  # Number of points on the flat plane
plane_size = 10.0  # Size of the plane (e.g., 10x10 units)
plane_height = 0.0  # The Z value for the flat plane

# Create a flat plane (a grid of points)
x_vals = np.linspace(-plane_size / 2, plane_size / 2, int(np.sqrt(num_points)))
y_vals = np.linspace(-plane_size / 2, plane_size / 2, int(np.sqrt(num_points)))
x_grid, y_grid = np.meshgrid(x_vals, y_vals)
z_vals = np.full_like(x_grid, plane_height)

# Flatten the grid to create a 1D array of points
plane_points = np.vstack((x_grid.flatten(), y_grid.flatten(), z_vals.flatten())).T

# Add noise to the plane
noise_level = 0.5  # Adjust this to control the amount of noise
noise = np.random.randn(plane_points.shape[0], 3) * noise_level
noisy_points = plane_points + noise

# Add additional random noisy points (points far from the plane)
num_noisy_points = 500
random_points = np.random.uniform(low=-plane_size / 2, high=plane_size / 2, size=(num_noisy_points, 3))

# Combine the noisy plane points with the random noisy points
all_points = np.vstack((noisy_points, random_points))

# Convert to Open3D point cloud
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(all_points)

# Visualize the point cloud
o3d.visualization.draw_geometries([point_cloud])
