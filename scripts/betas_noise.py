import numpy as np
from PIL import Image
import os
import pickle
import matplotlib.pyplot as plt

from utils import xy_array
from segmentation import Segmentation

betas_list = np.linspace(1, 10000, 20)
noise_std_list = np.linspace(0, 0.5, 20)

seeds_file_path = os.path.join('input', 'chien_', 'chien_10_seeds.pickle')
image_path = os.path.join('input', 'chien_', 'chien.png')
image_name = os.path.basename(image_path)
image = Image.open(image_path)
image_array = xy_array(np.array(image))

correct_segmentation_path = 'input/chien_/chien_got.png'
correct_segmentation_image = np.array(Image.open(correct_segmentation_path))

error_values = []

with open(seeds_file_path, 'rb') as seeds_file:
    seeds = pickle.load(seeds_file)
for beta in betas_list:
    for noise_std in noise_std_list:
        segmentation = Segmentation(image_array, beta, seeds, image_name)
        segmentation.add_noise(0, noise_std)
        segmentation.set_reference_segmentation(correct_segmentation_path)
        segmentation.solve()
        segmentation.build_segmentation_image()
        error_values.append(segmentation.compute_error())
        

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(betas_list, noise_std_list, error_values)
plt.show()
