#! /usr/bin/env python3

from numpy import linalg
import matplotlib.image as mpimg
from tkinter import *
from tkinter.filedialog import askopenfilename
import config


def interface():

    # Build canvas
    root = Tk()
    frame = Frame(root)
    canvas = Canvas(frame)
    canvas.grid(row=0, column=0, sticky=W)
    frame.pack(fill=BOTH, expand=1)

    file_path = askopenfilename(parent=root, title='Select an image.')
    print(f'opening file {file_path}')
    image = PhotoImage(file=file_path)
    canvas.create_image(0, 0, image=image, anchor="nw")
    canvas.config(scrollregion=canvas.bbox(ALL))

    solve_button = Button(root, text="Solve", command=root.quit)
    solve_button.pack()

    colours_list = Listbox(root)
    colours_list.pack()
    for colour in config.COLOURS_DIC.keys():
        colours_list.insert(END, colour)

    # Initialize variables
    seeds = []
    CURRENT_COLOUR = StringVar()

    def add_seed(event):
        cx, cy = canvas_coords(event, canvas)
        seeds.append({
            "x": cx,
            "y": cy,
            "colour": CURRENT_COLOUR.get()
        })
        canvas.create_oval(cx-config.OVAL_SIZE/2, cy-config.OVAL_SIZE/2, cx+config.OVAL_SIZE/2, cy +
                           config.OVAL_SIZE/2, width=2, fill=CURRENT_COLOUR.get())
        print(f'seeds = {seeds}')
        print("(%d, %d) / (%d, %d)" % (event.x, event.y, cx, cy))

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
