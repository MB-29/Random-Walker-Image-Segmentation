import numpy as np
from PIL import Image
import os
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as mtri

from utils import xy_array
from segmentation import Segmentation
import config

size = 10

seeds_file_path = os.path.join(config.SEEDS_PATH, 'dog_10_seeds.pickle')
image_path = os.path.join(config.IMAGES_PATH, 'dog.png')
target_segmentation_path = os.path.join(config.GOT_PATH, 'dog_got.png')

image_name = os.path.basename(image_path)
image = Image.open(image_path)
image_array = xy_array(np.array(image))
target_segmentation_image = np.array(Image.open(target_segmentation_path))

index = 1
std_list = np.linspace(0, 0.02, size)
betas_list = np.linspace(0, 5000, size)
error_values = []

with open(seeds_file_path, 'rb') as seeds_file:
    seeds = pickle.load(seeds_file)
for noise_std in std_list:
    segmentation = Segmentation(image_array, 1, seeds, image_name)
    segmentation.add_noise(0, noise_std)
    segmentation.set_reference_segmentation(target_segmentation_path)
    for beta in betas_list:
        print(f'index = {index}, beta = {beta}, std = {noise_std}')
        segmentation.beta = beta
        segmentation.solve()
        segmentation.build_segmentation_image()
        error_values.append(segmentation.compute_error())
        index += 1

X, Y = np.meshgrid(betas_list, std_list)
triang = mtri.Triangulation(np.reshape(
    X, [size*size]), np.reshape(Y, [size*size]))
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel(r'$\beta$')
ax.set_ylabel('noise')
ax.set_zlabel('error')
ax.set_zlim3d(0, 0.085)
ax.set_zticks([0.1])
ax.set_xticks([0, 5000])
ax.set_yticks([0, 0.02])

ax.plot_trisurf(triang, error_values, cmap='jet')

plt.show()
