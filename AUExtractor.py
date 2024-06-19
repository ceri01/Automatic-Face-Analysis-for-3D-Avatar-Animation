import struct
import asyncio
import socket
import numpy as np
from io import BytesIO
from PIL import Image
from feat import Detector

# SETUP
# detector used to acquire faces, landmarks and AUs
detector = Detector(face_model='faceboxes', landmark_model='mobilefacenet', au_model='xgb')

# socket setup
IP = '127.0.0.1'
PORT = 8052
socket_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# names of aus
AUsNames = [
    "AU1", "AU2", "AU4", "AU5", "AU6", "AU7", "AU9", "AU10",
    "AU11", "AU12", "AU14", "AU15", "AU17", "AU20", "AU23",
    "AU24", "AU25", "AU26", "AU28", "AU43"
]


async def mainLoop():
    while True:
        # Raw data -> first 8 bytes timestamp and other 8294400 frame bytes
        data = b''

        timestamp = b'' + socket_client.recv(8)
        # get all data from socket
        while len(data) < 8294400:
            data += socket_client.recv(4096)

        print(data)
        print("lunghezza finale: ", len(data))

        # from np.array of byte to PIL Image
        frame = generateNpArray(data)

        # from BGRA to BGR, remove opacity
        frame = frame[:, :, ::-1]

        # get aus (list of double)
        curr_aus = (await detectAus(frame))[0][0].tolist()

        print(curr_aus)

        if len(curr_aus[0]) > 0:
            # normalize aus
            aus_list = NormalizeData(curr_aus)

            # convert AUs into
            for au in range(len(aus_list)):
                aus_list[au] = fromDoubleToFloat(aus_list[au])

            socket_client.send(timestamp + b''.join(struct.pack('d', num) for num in aus_list))  # send to server
        await asyncio.sleep(0.01)


# Convert array of byte in np array, readable from py feat
def generateNpArray(frame):
    return np.array(Image.open(BytesIO(frame)))


# detect aus with py-feat
async def detectAus(frame):
    detected_face = detector.detect_faces(frame)
    detected_landmarks = detector.detect_landmarks(frame, detected_face)
    return detector.detect_aus(frame, detected_landmarks)


# normalize data in scale 0 to 100
def NormalizeData(data: list):
    return [int(val * 1000) / 10 for val in data]


# Convert Double (8 byte) into float (4 byte)
def fromDoubleToFloat(val):
    return struct.pack('>f', struct.unpack('>d', val)[0])


if __name__ == "__main__":
    socket_client.connect((IP, PORT))
    asyncio.run(mainLoop())
