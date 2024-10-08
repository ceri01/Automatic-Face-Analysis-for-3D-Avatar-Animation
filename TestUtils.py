import re
import libreface
import numpy as np
from PIL import Image


def catch_aus_feat(filename, detector):
    img = Image.open(filename)
    # Convert the PIL image into a NumPy array using numpy.array()
    frame = np.array(img)

    frame = frame[:, :, ::-1]

    detected_face = detector.detect_faces(frame)
    detected_landmarks = detector.detect_landmarks(frame, detected_face)
    aus = detector.detect_aus(frame, detected_landmarks)

    # aus contiene una lista contenente dei array multidimensionali di numpy, selezionando il primo elemento da aus
    # si ottiene il primo array multidimensionale, rappresentato come lista di liste.
    # accedendo con il secondo [0] si ottiene la lista di action unit
    if len(aus) >= 1 and len(aus[0]) >= 1:
        aus = aus[0][0].tolist()

        del aus[8]
        del aus[17]
        aus[17] = 0

        aus_list = normalize_aus(aus)
        return aus_list
    else:
        return []


def catch_aus_libreface(filename):
    curr_aus = libreface.get_au_intensities(filename)

    curr_aus["au_7_intensity"] = 0
    curr_aus["au_10_intensity"] = 0
    curr_aus["au_14_intensity"] = 0
    curr_aus["au_23_intensity"] = 0
    curr_aus["au_28_intensity"] = 0
    curr_aus["au_45_intensity"] = 0

    if len(curr_aus) > 0:
        aus_list = list(dict(sorted(curr_aus.items(), key= lambda x : extract_key_number(x[0]))).values())
        return normalize_aus(aus_list, 0, 5)
    else:
        return []


def extract_key_number(key):
    match = re.search(r'au_(\d+)_intensity', key)
    if match:
        return int(match.group(1))
    else:
        return float('inf')


# openFACS format
def normalize_aus(data, original_min=0, original_max=1):
    target_min = 0
    target_max = 5
    normalized = [((d - original_min) / (original_max - original_min)) * (target_max - target_min) + target_min for d in data]
    return normalized