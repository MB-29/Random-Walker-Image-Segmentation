#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 09:55:08 2019

@author: julesdelemotte
"""
import pickle
import networkx
import numpy as np
import itertools
import scipy
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy.random as rd
import os
from PIL import Image


from maths import get_neighbour_pixels, get_ordered_nodelist, pixel_norm_2
from config import COLOUR_RGB_MAP, COLOURS_DIC, OUTPUT_PATH, NOISE_PATH, GARBAGE_PATH,INPUT_PATH

def img_from_array(array,image_name):
    mpimg.imsave(image_name, array,format='png')

def build_White_picture_with_black_square_in_the_middle(size):
        
    # Computation of the image 
    image = np.zeros((size, size, 3))
    for i in range(size):
        for j in range(size):
            if i<size/4 or i>3*size/4:
                image[i][j] = COLOUR_RGB_MAP["white"]/255
            else:
                if j<size/4 or j>3*size/4:
                    image[i][j] = COLOUR_RGB_MAP["white"]/255
                else:
                    image[i][j] = COLOUR_RGB_MAP["black"]/255
                    
                    current =os.getcwd()
        
    # Black bounds
    for i in range(0,size,size-1):
        for j in range(size):
            image[i][j]=COLOUR_RGB_MAP["black"]/255
            image[j][i]=COLOUR_RGB_MAP["black"]/255
            
    # Save the current location 
    current = os.getcwd()
    
    # backup at the chosen place
    os.chdir(INPUT_PATH)
    mpimg.imsave("White_picture_with_black_square_in_the_middle.png", image,format='png')
    
    # Back to current location
    os.chdir(current)
    
def load_pickle(nom_fichier,chemin_enregistrement):
    
    # Saving current working folder
    current =os.getcwd()
    
    # backup at the chosen place
    os.chdir(chemin_enregistrement)
    retur=pickle.load(open(nom_fichier, 'rb'))
    os.chdir(current)
    
    return retur
