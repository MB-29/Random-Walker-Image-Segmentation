from tkinter import PhotoImage
from tkinter import *
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import networkx as nx
import itertools
from maths import *

from maths import xy_array, weight

image_path = 'o.png'
im = Image.open(image_path)
plt.imshow(im)

image_array = xy_array(np.array(im))

graph = build_weighted_graph(image_array)
print(
    f'Number of edges : {len(graph.edges)}, number of vertices : {len(graph.nodes())}')
laplacian = networkx.laplacian_matrix(graph)
print(laplacian)