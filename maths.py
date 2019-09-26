import numpy as np
import itertools
import networkx
import time
import os
import pickle

from config import COLOUR_RGB_MAP, BETA, INTENSITY_NORMALIZATION


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


def build_weighted_graph(image_array, beta):
    G = networkx.Graph()
    ny, nx = image_array.shape
    start_time = time.time()
    for x, y in itertools.product(range(nx), range(ny)):
        neighbours = get_neighbour_pixels(x, y, nx, ny)
        for pixel in neighbours:
            g, h = image_array[y][x], image_array[pixel[1]][pixel[0]]
            w = weight(float(g), float(h), beta)
            G.add_edge((x, y), pixel, weight=w)
    print("--- %s seconds ---" % (time.time() - start_time))
    return G


def get_neighbour_pixels(x, y, nx, ny):
    neighbours = []
    for pixel in [(x, y-1), (x-1, y)]:
        u, v = pixel
        if u >= 0 and v >= 0:
            neighbours.append((u, v))
    return neighbours


def get_ordered_nodelist(nodes_list, seeds_list):
    for seed in seeds_list[::-1]:
        print(f'seed {seed}')
        nodes_list.insert(0, nodes_list.pop(nodes_list.index(seed)))
    return nodes_list


def build_segmentation_image(nx, ny, pixel_colour_dic):
    image = np.zeros((ny, nx, 3))
    for i in range(ny):
        for j in range(nx):
            image[i][j] = COLOUR_RGB_MAP[pixel_colour_dic[(j, i)]]
    return image


def draw_contours(nx, ny, pixel_colour_dic):
    contours_array = np.zeros((ny, nx, 4))
    for x, y in itertools.product(range(nx), range(ny)):
        colour = pixel_colour_dic[(x, y)]
        neighbours = get_neighbour_pixels(x, y, nx, ny)
        for neighbour_pixel in neighbours:
            if pixel_colour_dic[neighbour_pixel] != colour:
                contours_array[y][x] = [255,0,0,1]
    return contours_array

def gaussian(g, h, beta):
    arg = -(g-h)*(g-h)*beta
    return np.exp(arg)

def pixel_norm_2(pixel):
    norm=0
    for i in pixel:
        norm+=i**2
    return np.sqrt(norm)/INTENSITY_NORMALIZATION

def load_pickle(nom_fichier,chemin_enregistrement):
    
    # Saving current working folder
    current =os.getcwd()
    
    # backup at the chosen place
    os.chdir(chemin_enregistrement)
    retur=pickle.load(open(nom_fichier, 'rb'))
    os.chdir(current)
    
    return retur


    