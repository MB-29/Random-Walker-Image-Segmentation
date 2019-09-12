import matplotlib.pyplot as plt
from maths import build_weighted_graph, order_nodelist
import networkx
import numpy as np
import scipy


def solve(seeds_dic, image_array):

    print('Starting resolution')

    K = len(seeds_dic.keys())
    seeds_coords_list = list(seeds_dic.keys())
    nx, ny = image_array.shape
    pixel_count = nx * ny

    print(f'Image dimensions : {image_array.shape}\nNumber of seeds : K={K}')

    # Build weighted graph and linear algebra objects
    print('building graph')
    graph = build_weighted_graph(image_array)
    print(
        f'Number of edges : {len(graph.edges)}, number of vertices : {len(graph.nodes())}')
    ordred_nodes = order_nodelist(list(graph), seeds_coords_list)
    print(f'ordered nodes : {ordred_nodes}')
    print(f'computing laplacian')
    laplacian = networkx.laplacian_matrix(graph, nodelist=ordred_nodes)
    laplacian_unseeded = laplacian[K:, K:]
    b_transpose = laplacian[K:, :K]

    unseeded_potentials_list = []
    for seed_index in range(K):
        seeds_vector = [0] * K
        seeds_vector[seed_index] = 1
        print(
            f'shapes :  {laplacian_unseeded.shape}, {(-b_transpose @ seeds_vector).shape}')
        unseeded_potentials = scipy.sparse.linalg.spsolve(
            laplacian_unseeded, -b_transpose @ seeds_vector)
        unseeded_potentials_list.append(unseeded_potentials)

    pixel_colour_dic = seeds_dic
    for pixel_index in range(0,pixel_count-K):
        pixel_coords = ordred_nodes[pixel_index]
        pixel_probabilities = [potentials[pixel_index] for potentials in unseeded_potentials_list]
        argmax_seed_index = np.argmax(pixel_probabilities)
        argmax_seed_coords = seeds_coords_list[argmax_seed_index]
        pixel_colour_dic.update({
            pixel_coords: seeds_dic[argmax_seed_coords]
        })
    print(pixel_colour_dic)


    # print(f'seeds = {seeds_coords_list}')
    # print(f'nodes = {list(graph)}')
    # print(f'ordered = {ordred_nodes}')
    # print(f'laplacian : {laplacian}')

    # plt.imshow(image_array)
    # plt.show()

    return
