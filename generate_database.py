#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 13:35:52 2019

@author: julesdelemotte
"""

import pickle
import networkx
import numpy as np
import itertools
import scipy
import matplotlib.pyplot as plt
import numpy.random as rd
import os

from maths import get_neighbour_pixels, get_ordered_nodelist, pixel_norm_2, gaussian
from config import COLOUR_RGB_MAP, COLOURS_DIC, OUTPUT_PATH, NOISE_PATH, GARBAGE_PATH, IMAGE_1
from constructor import load_pickle
from segmentation import Segmentation, Noise

def generate_noise_database(nom_pickle):
    
    # Importer l'object
    initial_segmentation=load_pickle(nom_pickle,OUTPUT_PATH+"/"+NOISE_PATH+"/"+nom_pickle)
    
    # On crée ensuite l'object pour toute les valeurs de bruit entre 0 et 0.5 de probabilité (au delà plus trop de sens)
    for i in np.arange(0,0.21,0.01):
        noised_segmentation=Noise(initial_segmentation.image_array.copy(), initial_segmentation.beta, initial_segmentation.seeds_dic,i)
        noised_segmentation.image_name=nom_pickle
        noised_segmentation.beta=100
        noised_segmentation.solve()
        noised_segmentation.build_segmentation_image()
        for i in range(noised_segmentation.nx):
            for j in range(noised_segmentation.ny):
                noised_segmentation.segmentation_image[i][j]=pixel_norm_2(noised_segmentation.segmentation_image[i][j])
        noised_segmentation.backup_pickle()
        
    print("Image bruité créé et segmentation effectué")
    
def test(nom_pickle):
    
    # Importer l'object
    initial_segmentation=load_pickle(nom_pickle,OUTPUT_PATH+"/"+NOISE_PATH+"/"+nom_pickle)
    
    # On crée ensuite l'object pour toute les valeurs de bruit entre 0 et 0.5 de probabilité (au delà plus trop de sens)
    noised_segmentation=Noise(initial_segmentation.image_array, initial_segmentation.beta, initial_segmentation.seeds_dic,0.20)
    noised_segmentation.image_name=nom_pickle
    noised_segmentation.beta=10
    noised_segmentation.solve()
    noised_segmentation.build_segmentation_image()
    for i in range(noised_segmentation.nx):
        for j in range(noised_segmentation.ny):
            noised_segmentation.segmentation_image[i][j]=pixel_norm_2(noised_segmentation.segmentation_image[i][j])
    noised_segmentation.backup_pickle()
    plt.imshow(noised_segmentation.image_array)
    plt.show()
    plt.imshow(noised_segmentation.segmentation_image)
    plt.show()
    print("Segmentation effectué")
    
generate_noise_database(IMAGE_1)

