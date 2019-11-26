import pickle
import networkx
import numpy as np
import itertools
import scipy
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy.random as rd
import os
import datetime
from PIL import Image

from utils import get_neighbour_pixels, get_ordered_nodelist, pixel_norm_2, gaussian, xy_array
from config import COLOUR_RGB_MAP, COLOURS_DIC, OUTPUT_PATH


class Segmentation:
    def __init__(self, image_array, beta, seeds_dic, image_name, reference_path=None):
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
        self.image_name = image_name
        self.segmenation_name = f'rw_{self.image_name}_{datetime.datetime.now()}'
        self.output_path = os.path.join(
            OUTPUT_PATH, self.segmenation_name+'.png')
        if reference_path:
            self.set_reference_segmentation(reference_path)
        self.error = 0

        print(
            f'Image dimensions : {self.image_array.shape}\nNumber of seeds : K={self.K}')
        print(f'seeds {self.seeds_dic}')

        # output directory
        if not os.path.isdir(OUTPUT_PATH):
            try:
                os.mkdir(OUTPUT_PATH)
            except Exception as e:
                print(e)

    def save_object(self):
        file_path = os.path.join(
            OUTPUT_PATH, f'{self.segmenation_name}.pickle')
        print(f'Saving segmentation object to {file_path}')
        with open(file_path, 'wb') as pickle_file:
            pickle.dump(self, pickle_file)

    def save_segmentation_image(self):
        file_path = os.path.join(
            OUTPUT_PATH, f'{self.image_name}_segmentation.pickle')
        print(f'Saving segmentation image to {file_path}')
        with open(file_path, 'wb') as pickle_file:
            pickle.dump(self.segmentation_image, pickle_file)

    def build_weighted_graph(self):
        for y, x in itertools.product(range(self.ny), range(self.nx)):
            neighbours = get_neighbour_pixels(x, y, self.nx, self.ny)
            for pixel in neighbours:
                g, h = self.image_array[y][x], self.image_array[pixel[1]][pixel[0]]
                w = self.weight_function(float(g), float(h), self.beta)
                self.graph.add_edge((x, y), pixel, weight=w)

    def build_linear_algebra(self):
        print('building graph')
        self.build_weighted_graph()
        self.ordered_nodes = get_ordered_nodelist(
            list(self.graph), list(self.seeds_dic.keys()))
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
            print(f'Solving system {seed_index+1} out of {self.K}')
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
        for y, x in itertools.product(range(self.ny), range(self.nx)):
            colour = self.pixel_colour_dic[(x, y)]
            neighbours = get_neighbour_pixels(x, y, self.nx, self.ny)
            for neighbour_pixel in neighbours:
                if self.pixel_colour_dic[neighbour_pixel] != colour:
                    contours_array[y][x] = [255, 0, 0, 1]
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
        plt.title(r'$\beta=$'+str(round(self.beta, 3)))
        plt.show()

    def plot_colours(self):
        if not self.solved:
            raise Exception('Impossible to plot segmentation before solving')
        self.build_segmentation_image()
        plt.title(r'$\beta=$'+str(round(self.beta, 3)))
        plt.imshow(self.segmentation_image)
        for seed in self.seeds_dic.keys():
            plt.plot(
                *seed, color=self.seeds_dic[seed], marker='o', markeredgecolor='yellow')
        plt.show()

    def segmentation_to_png(self, path=None):
        self.build_segmentation_image()
        output_path = self.output_path
        if path:
            output_path = path
        plt.imshow(self.segmentation_image)
        title = r'$\beta=$'+str(round(self.beta, 3))
        if self.error:
            title += f'\nerror = {round(self.error, 3)}'
        plt.title(title)
        for seed in self.seeds_dic.keys():
            plt.plot(
                *seed, color=self.seeds_dic[seed], marker='o', markeredgecolor='yellow')
        print(f'Saving colours image to {output_path}')
        plt.savefig(output_path)

    def contours_to_png(self, path=None):
        self.draw_contours()
        output_path = self.output_path
        if path:
            output_path = path
        plt.imshow(self.image_array, cmap='gray')
        plt.imshow(self.contours_array)
        title = r'$\beta=$'+str(round(self.beta, 3))
        if self.error:
            title += f'\nerror = {round(self.error, 3)}'
        plt.title(title)
        for seed in self.seeds_dic.keys():
            plt.plot(
                *seed, color=self.seeds_dic[seed], marker='o', markeredgecolor='yellow')
        print(f'Saving contours image to {output_path}')
        plt.savefig(output_path)

    def seeds_to_png(self, path=None):
        output_path = self.output_path
        if path:
            output_path = path
        plt.imshow(self.image_array, cmap='gray')
        print(f'seeds : {self.seeds_dic}')
        for seed in self.seeds_dic.keys():
            print(f'plotting seed {seed}')
            plt.plot(
                *seed, color=self.seeds_dic[seed], marker='o', markeredgecolor='yellow')
        print(f'Saving seeds image to {output_path}')
        plt.savefig(output_path)

    def compute_error(self, path=None):
        reference = self.reference
        if path:
            reference = xy_array(np.array(Image.open(path)))
        error = 0
        for y in range(self.ny):
            for x in range(self.nx):
                if not np.array_equal(
                        self.segmentation_image[y][x], reference[y][x]):
                    error += 1
        self.error = error/self.pixel_number
        return self.error

    def set_reference_segmentation(self, reference_path):
        reference_image = np.array(Image.open(reference_path))
        self.reference = reference_image

    def add_noise(self, mean, std):
        noisy_image = self.image_array + \
            np.random.normal(mean, std, self.image_array.shape)
        self.image_array = np.clip(noisy_image, 0, 255)
