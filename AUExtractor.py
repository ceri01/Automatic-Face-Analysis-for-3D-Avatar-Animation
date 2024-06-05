import struct
import asyncio
import sys

from feat import Detector
import socket

import AUExtractorGUI

# SETUP
# detector used to acquire faces, landmarks and AUs
detector = Detector(face_model='faceboxes', landmark_model='mobilefacenet', au_model='xgb')

# socket setup
IP = '127.0.0.1'
PORT = 8080
socket = socket.socket()

# names of aus
AUsNames = [
    "AU1", "AU2", "AU4", "AU5", "AU6", "AU7", "AU9", "AU10",
    "AU11", "AU12", "AU14", "AU15", "AU17", "AU20", "AU23",
    "AU24", "AU25", "AU26", "AU28", "AU43"
]


async def mainLoop(displayGui=False):
    while True:
        ausArray = bytearray()  # array ready to be sent
        # 8 -> timestamp
        # 8294400 -> frame size
        data = socket.recv(8 + 8294400)
        timestamp, frame = unpackData(data)
        frame = frame[:, :, 0:3]  # from BGRA to BGR, remove opacity
        curr_aus = await detectAus(frame)
        if len(curr_aus[0]) > 0:
            aus_list = NormalizeData(curr_aus[0][0].tolist())
            # convert AUs into byte
            for au in aus_list:
                ausArray.extend(struct.pack('f', au))
            socket.send(bytearray(timestamp) + ausArray)  # send to server
        await asyncio.sleep(0.01)


# Unpack received data, first 8 bytes are timestamp, other ones are frame
def unpackData(data):
    timestamp = data[:8]
    frame = data[8:]
    return timestamp, frame


# detect aus with py-feat
async def detectAus(frame):
    detected_face = detector.detect_faces(frame)
    detected_landmarks = detector.detect_landmarks(frame, detected_face)
    return detector.detect_aus(frame, detected_landmarks)


# normalize data for openFACS (used for previous tests)
def NormalizeData(data: list):
    normalized_data = [int(val * 1000) / 10 for val in data]
    return normalized_data


if __name__ == "__main__":
    displayGui = False
    if len(sys.argv) >= 2:
        displayGui = sys.argv[1]
    socket.connect((IP, PORT))
    AUExtractorGUI.OpenGUI()
    asyncio.run(mainLoop(displayGui))
