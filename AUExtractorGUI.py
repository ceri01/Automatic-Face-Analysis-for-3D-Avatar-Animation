import gi
import numpy as np
import matplotlib
import matplotlib.colors as mcolors
from matplotlib.collections import PatchCollection

# backand to rendering GUI
matplotlib.use('GTK4Agg')

import matplotlib.pyplot as plt


# enable plot interactive mode
plt.ion()

# PLOTS
# plot
plts, plst_axs = plt.subplots(2, 1)

# names of aus
au_names = [
    "AU1", "AU2", "AU4", "AU5", "AU6", "AU7", "AU9", "AU10",
    "AU12", "AU14", "AU15", "AU17", "AU20", "AU23", "AU25",
    "AU26", "AU28"
]

# Color gradient map from white to red
colorMap = mcolors.LinearSegmentedColormap.from_list("AU_value", ["white", "pink", "red"])

# square list
squares = []

# populate square
for index, value in enumerate(au_names):
    squares.append(plt.Rectangle(((index * 16), 1), 15, 15))
    plst_axs[1].text((index * 16) + 8, -2, au_names[index], ha="center", va="center", color="black", fontsize=10)

collection = PatchCollection(squares, edgecolors="black", cmap=colorMap, linewidth=1)
plst_axs[1].add_collection(collection)
plst_axs[1].set_ylim(0, 16)
plst_axs[1].set_xlim(0, (len(au_names) * 16))
plst_axs[1].set_aspect('equal')
plst_axs[1].axis('off')

plst_axs[0].set_aspect('equal')
plst_axs[0].axis('off')


def update_plot(data, frame):
    collection.set_array(np.array(data))
    plst_axs[0].imshow(frame)
    plts.canvas.draw_idle()
    plts.canvas.start_event_loop(0.001)


def OpenGUI():
    plt.show()

