import numpy as np
from PIL import Image
import os
import pickle
import matplotlib.pyplot as plt

from maths import xy_array
from segmentation import Segmentation

noise_std_list = np.linspace(0,0.5, 20)
seeds_file_path = os.path.join('input','rectangles_4_seeds.pickle')

image_path = os.path.join('input', 'rectangles.png')
image_name = os.path.splitext(image_path)[0]
image_name = os.path.basename(image_path)
image = Image.open(image_path)
image_array = xy_array(np.array(image))

with open(seeds_file_path, 'rb') as seeds_file:
    seeds = pickle.load(seeds_file)
index = 0
beta = 13

correct_segmentation_path = 'output/rectangles_segmentation.pickle'
with open(correct_segmentation_path, 'rb') as pickle_file:
    correct_segmentation_image = pickle.load(pickle_file)

for std in noise_std_list:
    file_path = f'output/rectangles_noise_{index}'
    segmentation = Segmentation(image_array, beta, seeds, image_name)
    segmentation.add_noise(0,std)
    segmentation.solve()
    segmentation.draw_contours()
    segmentation.build_segmentation_image()
    # segmentation.contours_to_png(path=file_path)

   
    error = segmentation.compute_error(correct_segmentation_image)

    plt.plot(std, error, marker='x', color='blue')

    index += 1
plt.show()