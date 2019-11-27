#! /usr/bin/env python3

from numpy import linalg
from tkinter import *
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
import numpy as np
import logging
from collections import OrderedDict
import os
import pickle
import time
import json

import config
from utils import xy_array
from segmentation import Segmentation


def interface():

    root = Tk()
    frame = Frame(root)

    # Choose image
    image_path = askopenfilename(parent=root, title='Select an image.')
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    print(f'opening image {image_path}')
    image = Image.open(image_path)
    tk_image = ImageTk.PhotoImage(image)
    height, width = tk_image.height(), tk_image.width()

    # Build canvas
    canvas = Canvas(frame, width=width+100, height=height+100)
    canvas.grid()
    frame.pack()
    beta_entry = Entry(root)
    beta_entry.pack()
    beta_entry.insert(0, 'Beta parameter')

    canvas.create_image(0, 0, image=tk_image, anchor="nw")
    canvas.config(scrollregion=canvas.bbox(ALL))
    image_array = xy_array(np.array(image))

    save_segmentation = IntVar()
    save_segmentation_button = Checkbutton(
        root, text="Save segmentation", var=save_segmentation)
    save_segmentation_button.pack()

    draw_contours = IntVar()
    draw_contours_button = Checkbutton(
        root, text="Draw contours", var=draw_contours)
    draw_contours_button.pack()

    def on_solve():
        beta_parameter = float(beta_entry.get())
        segmentation = Segmentation(
            image_array, beta_parameter, seeds, image_name)
            
        segmentation.solve()

        if save_segmentation.get():
            segmentation.build_segmentation_image()
            segmentation.save_segmentation_image()

        if draw_contours.get():
            segmentation.plot_contours()

        segmentation.build_segmentation_image()
        segmentation.plot_colours()

    def save_seeds():
        image_directory_path = os.path.dirname(image_path)
        seeds_file_name = f'{image_name}_{len(seeds.keys())}_seeds.pickle'
        seeds_file_path = os.path.join(image_directory_path, seeds_file_name)
        print(f'Saving seeds to {seeds_file_path}')
        with open(seeds_file_path, 'wb') as pickle_file:
            pickle.dump(seeds, pickle_file)

    solve_button = Button(root, text="Solve", command=on_solve)
    solve_button.pack()

    save_seeds_button = Button(root, text="Save seeds", command=save_seeds)
    save_seeds_button.pack()

    solve_button.pack()

    colours_list = Listbox(root)
    colours_list.pack()
    for colour in config.COLOURS_DIC.keys():
        colours_list.insert(END, colour)

    # Initialize variables
    seeds = OrderedDict()
    CURRENT_COLOUR = StringVar()
    seed_ovals = []

    def add_seed(event):
        if not CURRENT_COLOUR.get():
            print('No colour selected !')
            return
        x, y = canvas_coords(event, canvas)
        seeds.update({
            (x, y): CURRENT_COLOUR.get()
        })
        last_seed = canvas.create_oval(x-config.OVAL_SIZE/2, y-config.OVAL_SIZE/2, x+config.OVAL_SIZE/2, y +
                                       config.OVAL_SIZE/2, width=2, fill=CURRENT_COLOUR.get())
        seed_ovals.append(last_seed)
        print(f'New {CURRENT_COLOUR.get()} seed added : {[x,y]}')

    def remove_seed(event):
        seeds.pop(next(reversed(seeds)))
        canvas.delete(seed_ovals.pop(len(seed_ovals) - 1))

    def select_colour(event):
        CURRENT_COLOUR.set(colours_list.get(colours_list.curselection()))
        print(f'current colour = {CURRENT_COLOUR.get()}')

    canvas.bind("<ButtonPress-1>", add_seed)
    canvas.bind("<ButtonPress-2>", remove_seed)
    colours_list.bind("<<ListboxSelect>>", select_colour)
    solve_button.bind

    root.mainloop()

# Transform event coordinates into canvas coordinates


def canvas_coords(event, canvas):
    return (canvas.canvasx(event.x), canvas.canvasy(event.y))


if __name__ == "__main__":
    interface()
