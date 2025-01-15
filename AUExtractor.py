import struct
import asyncio
import socket

import numpy as np
from feat import Detector

# SETUP
# detector used to acquire faces, landmarks and AUs
detector = Detector(face_model='faceboxes', landmark_model='mobilefacenet', au_model='xgb')

# socket setup
IP = '127.0.0.1'
PORT = 8053
socket_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
socket.setdefaulttimeout(3)

IP_DEST = '' # insert ip
PORT_DEST = 12345
dest_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Constants
NUM_AUS = 20
H = 720
W = 1280
C = 3

async def main_loop():
    while True:
        data = b''
        frame_size = H * W * C  # 2.764.800
        while len(data) < frame_size:
            rec = socket_client.recv(4096)
            if not rec:
                return
            data += rec

        # Create np.array
        frame = generate_np_array(data)

        # get aus
        curr_aus = await detect_aus(frame)

        # Create list of AUS if face is detected
        aus_in_byte = b''
        if len(curr_aus[0]) > 0:
            aus_list = normalize_data(curr_aus[0][0].tolist())
        else:
            # although create list of zeros
            aus_list = [0]*NUM_AUS

        # Pack list of AUS (20 AUS * 4 Bytes)
        for aus_val in aus_list:
            aus_in_byte += struct.pack('I', aus_val)

        # Send AUS to client
        dest_client.sendall(aus_in_byte)


# Convert array of byte in np array, readable from py feat
def generate_np_array(frame):
    return np.frombuffer(frame, dtype='uint8').reshape((H, W, C), order='C')


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
    dest_client.connect((IP_DEST, PORT_DEST))
    asyncio.run(main_loop())
