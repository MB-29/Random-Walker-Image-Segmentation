#! /usr/bin/env python3

from numpy import linalg
from tkinter import *
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
import numpy as np
import logging


import config
from solve import solve
from maths import xy_array

def interface():

    # Build canvas
    root = Tk()
    frame = Frame(root)
    canvas = Canvas(frame)
    canvas.grid(row=0, column=0, sticky=W)
    frame.pack()
    beta_entry = Entry(root)
    beta_entry.pack()

    # Open image
    file_path = askopenfilename(parent=root, title='Select an image.')
    print(f'opening file {file_path}')
    image = Image.open(file_path)
    tk_image = ImageTk.PhotoImage(image)
    canvas.create_image(0, 0, image=tk_image, anchor="nw")
    canvas.config(scrollregion=canvas.bbox(ALL))
    image_array = xy_array(np.array(image))

    def on_solve():
        beta = float(beta_entry.get())
        solve(seeds, image_array, beta=beta)
        root.quit()

    solve_button = Button(root, text="Solve", command=on_solve)
    solve_button.pack()

    colours_list = Listbox(root)
    colours_list.pack()
    for colour in config.COLOURS_DIC.keys():
        colours_list.insert(END, colour)

    # Initialize variables
    seeds = {}
    CURRENT_COLOUR = StringVar()

    def add_seed(event):
        if not CURRENT_COLOUR.get():
            print('No colour selected !')
            return
        x, y = canvas_coords(event, canvas)
        seeds.update({
            (x,y): CURRENT_COLOUR.get()
            })
        canvas.create_oval(x-config.OVAL_SIZE/2, y-config.OVAL_SIZE/2, x+config.OVAL_SIZE/2, y +
                           config.OVAL_SIZE/2, width=2, fill=CURRENT_COLOUR.get())
        print(f'New seed added : {[x,y]}')

    def select_colour(event):

        CURRENT_COLOUR.set(colours_list.get(colours_list.curselection()))
        print(f'current colour = {CURRENT_COLOUR.get()}')

    canvas.bind("<ButtonPress-1>", add_seed)
    colours_list.bind("<<ListboxSelect>>", select_colour)
    solve_button.bind

    root.mainloop()

# Transform event coordinates into canvas coordinates


def canvas_coords(event, canvas):
    return (canvas.canvasx(event.x), canvas.canvasy(event.y))


if __name__ == "__main__":
    interface()


