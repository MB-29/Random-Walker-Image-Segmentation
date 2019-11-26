import numpy as np
from PIL import Image
import os
import pickle
import matplotlib.pyplot as plt
from collections import OrderedDict
from scipy import optimize

from utils import xy_array
from segmentation import Segmentation

seeds_file_path = os.path.join('input', 'chien_', 'chien_42_seeds.pickle')

image_path = os.path.join('input', 'chien_', 'chien.png')
image_name = os.path.splitext(image_path)[0]
image_name = os.path.basename(image_path)
image = Image.open(image_path)
image_array = xy_array(np.array(image))

with open(seeds_file_path, 'rb') as seeds_file:
    seeds = pickle.load(seeds_file)
index = 0
beta = 1300
colour = 'red'

correct_segmentation_path = 'input/chien_/chien_got.png'

while len(seeds) > 1:
    if len(seeds) % 10 == 0:
        plt.clf()
        segmentation = Segmentation(image_array, beta, seeds,
                            image_name, reference_path=correct_segmentation_path)
        segmentation.seeds_to_png(path=f'input/chien_/chien_{len(seeds)}_seeds.png')
    # segmentation.solve()
    # segmentation.build_segmentation_image()
    # error = segmentation.compute_error()
    # plt.plot(len(seeds), error, marker='x', color='blue')
    seeds.popitem()
# plt.show()

    
