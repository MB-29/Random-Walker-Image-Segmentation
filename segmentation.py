import pickle
import networkx
import numpy as np
import itertools
import scipy
import matplotlib.pyplot as plt

from maths import get_neighbour_pixels, get_ordered_nodelist
from config import COLOUR_RGB_MAP, COLOURS_DIC

def gaussian(g, h, beta):
    arg = -(g-h)*(g-h)*beta
    return np.exp(arg)

class Segmentation:
    def __init__(self, image_array, beta, seeds_dic):
        self.image_array = image_array
        self.ny, self.nx = image_array.shape
        self.pixel_number = self.nx * self.ny
        self.beta = beta
        self.seeds_dic = seeds_dic
        self.pixel_colour_dic = self.seeds_dic.copy()
        self.graph = networkx.Graph()
        self.weight_function = gaussian
        self.K = len(self.seeds_dic.keys())
        self.solved = False

        print(f'Image dimensions : {self.image_array.shape}\nNumber of seeds : K={self.K}')

    def save(self):
        file_path = f'segmentation_objects/segmentation'
        print(f'Saving segmentation object to {file_path}')
        file = open(file_path, 'wb')
        pickle.dump(self, file)

    def build_weighted_graph(self):
        for x, y in itertools.product(range(self.nx), range(self.ny)):
            neighbours = get_neighbour_pixels(x, y, self.nx, self.ny)
            for pixel in neighbours:
                g, h = self.image_array[y][x], self.image_array[pixel[1]][pixel[0]]
                w = self.weight_function(float(g), float(h), self.beta)
                self.graph.add_edge((x, y), pixel, weight=w)

    def build_linear_algebra(self):
        print('building graph')
        self.build_weighted_graph()
        self.ordered_nodes = get_ordered_nodelist(list(self.graph), list(self.seeds_dic.keys()))
        print('computing laplacian')
        self.laplacian = networkx.laplacian_matrix(
            self.graph, nodelist=self.ordered_nodes, weight='weight')
        print('extracting sub-matrices')
        self.laplacian_unseeded = self.laplacian[self.K:, self.K:]
        self.b_transpose = self.laplacian[self.K:, :self.K]
    
    def solve_linear_systems(self):
        print('solving linear systems')
        unseeded_potentials_list = []
        for seed_index in range(self.K):
            print(f'Solving system {seed_index} out of {self.K-1}')
            seeds_vector = [0] * self.K
            seeds_vector[seed_index] = 1
            unseeded_potentials = scipy.sparse.linalg.spsolve(
                self.laplacian_unseeded, -self.b_transpose @ seeds_vector)
            unseeded_potentials_list.append(unseeded_potentials)
        return unseeded_potentials_list
    
    def assign_max_likelihood(self, unseeded_potentials_list):
        print('Assigning maximum likelihood seed')
        for pixel_index in range(self.K, self.pixel_number):
            pixel_coords = self.ordered_nodes[pixel_index]
            pixel_probabilities = [potentials[pixel_index - self.K]
                                for potentials in unseeded_potentials_list]
            argmax_seed_index = np.argmax(pixel_probabilities)
            argmax_seed_coords = list(self.seeds_dic.keys())[argmax_seed_index]
            self.pixel_colour_dic.update({
                pixel_coords: self.seeds_dic[argmax_seed_coords]
            })
    def build_segmentation_image(self):
        image = np.zeros((self.ny, self.nx, 3))
        for i in range(self.ny):
            for j in range(self.nx):
                image[i][j] = COLOUR_RGB_MAP[self.pixel_colour_dic[(j, i)]]
        self.segmentation_image = image
        return image
    
    def draw_contours(self):
        contours_array = np.zeros((self.ny, self.nx, 4))
        for x, y in itertools.product(range(self.nx), range(self.ny)):
            colour = self.pixel_colour_dic[(x, y)]
            neighbours = get_neighbour_pixels(x, y, self.nx, self.ny)
            for neighbour_pixel in neighbours:
                if self.pixel_colour_dic[neighbour_pixel] != colour:
                    contours_array[y][x] = [255,0,0,1]
        self.contours_array = contours_array
        return contours_array
    
    def solve(self):
        self.solved = True
        self.build_weighted_graph()
        self.build_linear_algebra()
        unseeded_potentials_list = self.solve_linear_systems()
        self.assign_max_likelihood(unseeded_potentials_list)
        return self.pixel_colour_dic
    
    def plot_contours(self):
        if not self.solved:
            raise Exception('Impossible to plot segmentation before solving')
        self.draw_contours()
        plt.imshow(self.image_array, cmap='gray')
        plt.imshow(self.contours_array)
        plt.show()