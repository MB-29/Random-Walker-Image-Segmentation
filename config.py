import numpy as np
import os

OVAL_SIZE = 10
COLOURS_DIC = {
    "white": 0,
    "black": 1,
    "red": 2,
    "green": 3,
    "blue": 4
}
COLOUR_RGB_MAP = {
    "white": [255, 255, 255],
    "black": [0, 0, 0],
    "red": [255, 0, 0],
    "green": [0, 255, 0],
    "blue": [0, 0, 255]
}
INTENSITY_NORMALIZATION = np.sqrt(3) * 255

SEEDS_PATH = os.path.join('input', 'seeds')
IMAGES_PATH = os.path.join('input', 'images')
GOT_PATH = os.path.join('input', 'got')
OUTPUT_PATH = 'output'

MAX_INTERFACE_WIDTH = 400
