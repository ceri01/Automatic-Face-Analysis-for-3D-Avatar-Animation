import struct
import asyncio
import socket
import sys
from sys import stderr

import numpy as np
from feat import Detector

# SETUP
# detector used to acquire faces, landmarks and AUs
detector = Detector(face_model='faceboxes', landmark_model='mobilefacenet', au_model='xgb')

# socket setup
IP = '127.0.0.1'
PORT = 8052
socket_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
socket.setdefaulttimeout(3)

# names of aus (sent in this order)
AUsNames = [
    "AU1", "AU2", "AU4", "AU5", "AU6", "AU7", "AU9", "AU10",
    "AU11", "AU12", "AU14", "AU15", "AU17", "AU20", "AU23",
    "AU24", "AU25", "AU26", "AU28", "AU43"
]


async def main_loop():
    while True:
        # Raw data -> first 8 bytes timestamp and other 8294400 frame bytes
        data = b''

        # get all data from socket
        try:
            while len(data) < 8294400:
                rec = socket_client.recv(4096)
                data += rec
                if len(rec) <= 0:
                    break

        except socket.timeout:
            sys.stderr("Socket timeout error!")
            continue

        try:
            # from np.array of byte to PIL Image
            frame = generate_np_array(data)

            # from BGRA to BGR, remove opacity
            frame = frame[:, :, ::-1]

            # get aus (list of double)
            curr_aus = await detect_aus(frame)

            if len(curr_aus[0]) > 0:
                # normalize aus
                aus_list = normalize_data(curr_aus[0][0].tolist())

                aus_in_byte = b''
                for aus in aus_list:
                    aus_in_byte += struct.pack('I', aus)

                socket_client.send(aus_in_byte)  # send to server

        except:
            sys.stderr("Error occurs in frame processing!")
            continue

        await asyncio.sleep(0.01)


# Convert array of byte in np array, readable from py feat
def generate_np_array(frame):
    return np.frombuffer(frame, dtype='uint8').reshape((1080, 1920, 4), order='C')


# detect aus with py-feat
async def detect_aus(frame):
    detected_face = detector.detect_faces(frame)
    detected_landmarks = detector.detect_landmarks(frame, detected_face)
    return detector.detect_aus(frame, detected_landmarks)


# normalize data in scale 0 to 100
def normalize_data(data: list):
    original_max = 1
    original_min = 0
    target_min = 0
    target_max = 100

    normalized = [int(((d - original_min) / (original_max - original_min)) * (target_max - target_min) + target_min) for
                  d in data]
    return normalized


if __name__ == "__main__":
    socket_client.connect((IP, PORT))
    asyncio.run(main_loop())
