import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

def get_image_texture(num):

    np.random.seed(0)

    for i in range(num):
        random_color = np.random.rand(3)
        image = np.ones((128, 128, 3)) * random_color
        plt.imsave(f"{i}.png", image)

    textures = []
    for i in range(num):
        texture = o3d.io.read_image(f"{i}.png")
        textures.append(texture)    

    return textures

if __name__ == "__main__":
    textures = get_image_texture(10)
    print(textures)