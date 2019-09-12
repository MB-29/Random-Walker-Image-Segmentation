import numpy as np
import itertools
import networkx
from config import COLOUR_RGB_MAP


def xy_array(array):
    if len(array.shape) == 2:
        return array
    nx, ny, m = array.shape
    print(f'(nx, ny, m) ={array.shape}')
    result = np.zeros([nx, ny])
    for i in range(nx):
        for j in range(ny):
            value = 0
            for k in range(m-1):
                value += (array[i][j][k]*1.0/255) ** 2
            result[i][j] = np.sqrt(value)
    return result


def weight(u, v):
    arg = -(u-v)*(u-v)
    return np.exp(arg)


def build_weighted_graph(image_array):
    G = networkx.Graph()
    nx, ny = image_array.shape

    for i, j in itertools.product(range(nx), range(ny)):
        for k, l in itertools.product(range(nx), range(ny)):
            distance = abs(i-k) + abs(j-l)
            if distance > 0 and distance < 2:
                u, v = image_array[i][j], image_array[k][l]
                G.add_edge((i, j), (k, l), weight=weight(float(u), float(v)))
    return G


def order_nodelist(nodes_list, seeds_list):
    for seed in seeds_list:
        nodes_list.insert(0, nodes_list.pop(nodes_list.index(seed)))
    return nodes_list

def build_segmentation_image(nx, ny, pixel_colour_dic):
    image = np.zeros((nx, ny, 3))
    for i in range(nx):
        for j in range(ny):
            image[i][j] = COLOUR_RGB_MAP[pixel_colour_dic[(i,j)]]
    return image
