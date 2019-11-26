import numpy as np
import itertools
import networkx
import time
import os
import pickle

from config import COLOUR_RGB_MAP, INTENSITY_NORMALIZATION, TIMINGS_PATH


def xy_array(array):
    if len(array.shape) == 2:
        return array
    ny, nx, m = array.shape
    print(f'(ny, nx, m) ={array.shape}')
    result = np.zeros([ny, nx])
    for i in range(ny):
        for j in range(nx):
            square_intensity = 0
            for k in range(m-1):
                square_intensity += array[i][j][k] ** 2
            result[i][j] = np.sqrt(square_intensity)/INTENSITY_NORMALIZATION
    return result


def weight(g, h, beta):
    arg = -(g-h)*(g-h)*beta
    return np.exp(arg)


def get_neighbour_pixels(x, y, nx, ny):
    neighbours = []
    for pixel in [(x, y-1), (x-1, y)]:
        u, v = pixel
        if u >= 0 and v >= 0:
            neighbours.append((u, v))
    return neighbours


def get_ordered_nodelist(nodes_list, seeds_list):
    for seed in seeds_list[::-1]:
        nodes_list.insert(0, nodes_list.pop(nodes_list.index(seed)))
    return nodes_list


def gaussian(g, h, beta):
    arg = -(g-h)*(g-h)*beta
    return np.exp(arg)


def pixel_norm_2(pixel):
    norm = 0
    for i in pixel:
        norm += i**2
    return np.sqrt(norm)/INTENSITY_NORMALIZATION


def record_time(nx, ny, K, timing):
    with open(TIMINGS_PATH, 'a') as timings_file:
        timings_file.write(f'{nx*ny} {K} {timing}\n')

