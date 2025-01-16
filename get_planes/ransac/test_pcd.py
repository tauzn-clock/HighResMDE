import numpy as np
import open3d as o3d
import random

def get_plane(R, EPSILON):
    # Parameters for the flat plane
    num_points = int(R//EPSILON)  # Number of points on the flat plane
    plane_size = R  # Size of the plane (e.g., 10x10 units)
    plane_height = 0.0  # The Z value for the flat plane

    # Create a flat plane (a grid of points)
    x_vals = np.linspace(-plane_size / 2, plane_size / 2, num_points)
    y_vals = np.linspace(-plane_size / 2, plane_size / 2, num_points)
    x_grid, y_grid = np.meshgrid(x_vals, y_vals)
    z_vals = np.full_like(x_grid, plane_height)

    # Flatten the grid to create a 1D array of points
    plane_points = np.vstack((x_grid.flatten(), y_grid.flatten(), z_vals.flatten())).T

    # Add noise to the plane
    noise_range = 3
    noise = np.random.randint(-noise_range, noise_range, [plane_points.shape[0], 3]) * EPSILON
    noisy_points = plane_points #+ noise

    # Add additional random noisy points (points far from the plane)
    num_noisy_points = (num_points**2) * 3
    random_points = np.random.randint(low=-num_points/2, high=num_points/ 2, size=(num_noisy_points, 3)) * EPSILON

    # Combine the noisy plane points with the random noisy points
    all_points = np.vstack((noisy_points, random_points))

    # Convert to Open3D point cloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(all_points)

    return point_cloud