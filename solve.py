import matplotlib.pyplot as plt
import networkx
import numpy as np
import scipy

from maths import build_weighted_graph, get_ordered_nodelist, build_segmentation_image
from config import BETA

def solve(seeds_dic, image_array, beta=BETA):

    print(f'Starting resolution')
    print(f'seeds : {seeds_dic}')

    K = len(seeds_dic.keys())
    seeds_coords_list = list(seeds_dic.keys())
    print(f'seeds_coords : {seeds_coords_list}')
    ny, nx = image_array.shape
    pixel_number = nx * ny

    print(f'Image dimensions : {image_array.shape}\nNumber of seeds : K={K}')

    # Build weighted graph and linear algebra objects
    print('building graph')
    graph = build_weighted_graph(image_array, beta=beta)
    ordered_nodes = get_ordered_nodelist(list(graph), seeds_coords_list)
    print('computing laplacian')
    laplacian = networkx.laplacian_matrix(graph, nodelist=ordered_nodes, weight='weight')
    print('extracting sub-matrices')
    laplacian_unseeded = laplacian[K:, K:]
    b_transpose = laplacian[K:, :K]

    # Solve linear system for every initial condition
    print('solving linear systems')
    unseeded_potentials_list = []
    for seed_index in range(K):
        seeds_vector = [0] * K
        seeds_vector[seed_index] = 1
        unseeded_potentials = scipy.sparse.linalg.spsolve(
            laplacian_unseeded, -b_transpose @ seeds_vector)
        unseeded_potentials_list.append(unseeded_potentials)

    # For each pixel choose maximum likelihood seed
    print('Assigning maximum likelihood seed')
    pixel_colour_dic = seeds_dic
    for pixel_index in range(K,pixel_number):
        pixel_coords = ordered_nodes[pixel_index]
        pixel_probabilities = [potentials[pixel_index - K] for potentials in unseeded_potentials_list]
        argmax_seed_index = np.argmax(pixel_probabilities)
        argmax_seed_coords = seeds_coords_list[argmax_seed_index]
        pixel_colour_dic.update({
            pixel_coords: seeds_dic[argmax_seed_coords]
        })
    
    # Build output
    print('Building output')
    segmentation_image = build_segmentation_image(nx, ny, pixel_colour_dic)
    plt.imshow(segmentation_image)
    plt.show()

    return
