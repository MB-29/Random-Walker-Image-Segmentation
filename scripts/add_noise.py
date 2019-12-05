import numpy as np
from PIL import Image
import os
import pickle
import matplotlib.pyplot as plt

from utils import xy_array

noise_std_list = np.linspace(0,0.5, 5)
image_path = os.path.join('input', 'palaiseau.png')
image_name = os.path.splitext(image_path)[0]
image_name = os.path.basename(image_path)
image = Image.open(image_path)
image_array = xy_array(np.array(image))

index = 0
for std in noise_std_list:
    noisy_image = image_array + np.random.normal(0, std, image_array.shape)
    image_array = np.clip(noisy_image, 0, 255)
    plt.imshow(image_array)
    output_path = f'output/palaiseau_noise_{index}'
    plt.savefig(output_path)
    index += 1
