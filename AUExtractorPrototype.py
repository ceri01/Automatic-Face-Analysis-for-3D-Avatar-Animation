import gi
import numpy as np
import matplotlib
import matplotlib.colors as mcolors
import asyncio
from matplotlib.collections import PatchCollection
from freenect2 import Device, FrameType
from feat import Detector
from openFACS.python.openFACS import sendAUS

# backand to rendering GUI
matplotlib.use('GTK4Agg')

import matplotlib.pyplot as plt

#SETUP
# detector used to acquire faces, landmarks and AUs
detector = Detector(face_model='faceboxes', landmark_model='mobilefacenet', au_model='xgb')

# used to control signals from kinect
device = Device()

# set required version of Gtk for gi
gi.require_version('Gtk', '4.0')

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


def main_loop():
    plt.show()
    asyncio.run(captureFrames())


async def detectAus(frame):
    detected_face = detector.detect_faces(frame)
    detected_landmarks = detector.detect_landmarks(frame, detected_face)
    return detector.detect_aus(frame, detected_landmarks)


async def captureFrames():
    device.start()
    try:
        while True:
            type_, frame = device.get_next_frame()
            if FrameType.Color is type_:
                frame = frame.to_array()[:720, :1280, 0:3]
                frame = frame[:, :, ::-1]
                curr_aus = await detectAus(frame)
                if len(curr_aus[0]) > 0:
                    aus_list = NormalizeData(formatData(curr_aus[0][0].tolist()))
                    sendAUS(aus_list, 0.005)
                    update_plot(aus_list, frame)
            await asyncio.sleep(0.01)
    finally:
        device.stop()


def formatData(data: list):
    del data[8]
    del data[14]
    data[17] = 0
    return data


def NormalizeData(data: list):
    return [5 * el for el in data]


if __name__ == "__main__":
    main_loop()
