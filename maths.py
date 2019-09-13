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
        neighbours = get_neighbour_pixels(i,j, nx, ny)
        for pixel in neighbours:
            print(f'(i,j) (k,l) = {(i,j)}, {pixel}')
            u, v = image_array[i][j], image_array[pixel[0]][pixel[1]]
            G.add_edge((i, j), pixel, weight=weight(float(u), float(v)))
    return G

def get_neighbour_pixels(i,j,nx,ny):
    neighbours = []
    for pixel in [(i, j-1), (i-1, j), (i, j+1), (i, j+1)]:
        x, y = pixel
        if x >= 0 and x < nx and y >= 0 and y < ny:
            neighbours.append((x,y))
    return neighbours

def order_nodelist(nodes_list, seeds_list):
    for seed in seeds_list:
        nodes_list.insert(0, nodes_list.pop(nodes_list.index(seed)))
    return nodes_list


def build_segmentation_image(nx, ny, pixel_colour_dic):
    image = np.zeros((nx, ny, 3))
    for i in range(nx):
        for j in range(ny):
            image[i][j] = COLOUR_RGB_MAP[pixel_colour_dic[(i, j)]]
    return image
