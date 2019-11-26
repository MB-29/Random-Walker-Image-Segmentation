import numpy as np
from PIL import Image
import os
import pickle
import matplotlib.pyplot as plt

from utils import xy_array
from segmentation import Segmentation

betas_list = np.linspace(1200, 10000, 20)
seeds_file_path = os.path.join('input','chien_', 'chien_42_seeds.pickle')

image_path = os.path.join('input', 'chien_','chien.png')
image_name = os.path.splitext(image_path)[0]
image_name = os.path.basename(image_path)
image = Image.open(image_path)
image_array = xy_array(np.array(image))

with open(seeds_file_path, 'rb') as seeds_file:
    seeds = pickle.load(seeds_file)
index = 0
betas_list=[1300]
for beta in betas_list:
    file_path = f'output/{image_name}_{index}.png'
    segmentation = Segmentation(image_array, beta, seeds, image_name)
    segmentation.solve()
    segmentation.build_segmentation_image()

    correct_segmentation_path = 'input/chien_/chien_got.png'
    correct_segmentation_image = np.array(Image.open(correct_segmentation_path))
    segmentation.set_reference_segmentation(correct_segmentation_path)
    error = segmentation.compute_error()
    segmentation.contours_to_png()
    index += 1
plt.show()
