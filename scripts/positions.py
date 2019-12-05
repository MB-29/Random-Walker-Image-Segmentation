import numpy as np
from PIL import Image
import os
import pickle
import matplotlib.pyplot as plt
from collections import OrderedDict
from scipy import optimize

from utils import xy_array
from segmentation import Segmentation
import config

# Load input

seeds_file_path = os.path.join(config.SEEDS_PATH, 'dog_10_seeds.pickle')
image_path = os.path.join(config.IMAGES_PATH, 'chien.png')
target_segmentation_path = os.path.join(config.GOT_PATH, 'chien_got.png')

image_name = os.path.splitext(image_path)[0]
image_name = os.path.basename(image_path)
image = Image.open(image_path)
image_array = xy_array(np.array(image))

with open(seeds_file_path, 'rb') as seeds_file:
    initial_seeds = pickle.load(seeds_file)
index = 0
beta = 1000
colour = 'red'

# error function
def error(coords_list):
    seeds = fixed_seeds.copy()
    print(f'coordinate list : {coords_list}')
    print(f'seeds : {seeds}')
    for index in range(len(coords_list)//2):
        coordinate = (int(coords_list[2*index]), int(coords_list[2*index+1]))
        seeds.update({coordinate: colour})
    segmentation = Segmentation(
        image_array, beta, seeds, image_name, reference_path=target_segmentation_path)
    segmentation.solve()
    segmentation.build_segmentation_image()
    return segmentation.compute_error()

# integer constraint
cons = {'type': 'eq', 'fun': lambda x: max(
    [x[i]-int(x[i]) for i in range(len(x))])}

fixed_seeds = {key: value for key,
               value in initial_seeds.items() if value == 'white'}
initial_variable_seeds = {key: value for key,
                 value in initial_seeds.items() if value == 'red'}
initial_coords = list(initial_variable_seeds.keys())
print(f'initial coords = {initial_coords}')

optimal_coords = optimize.minimize(
error, initial_coords, method='nelder-mead', constraints=cons, options={'xtol': 1e-3, 'disp': True})

optimal_seeds = fixed_seeds
for index in range(len(optimal_coords)//2):
    coordinate = (int(optimal_coords[2*index]), int(optimal_coords[2*index+1]))
    optimal_seeds.update({coordinate: colour})

# optimal segmentation
segmentation = Segmentation(image_array, beta, optimal_seeds,
                            image_name, reference_path=target_segmentation_path)
segmentation.solve()
segmentation.build_segmentation_image()
segmentation.compute_error()
segmentation.contours_to_png()
