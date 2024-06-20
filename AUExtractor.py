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
PORT = 8054
socket_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
socket.setdefaulttimeout(3)

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
        try:
            while len(data) < 8294400:
                print(".", end='')
                rec = socket_client.recv(1024)
                data += rec
                if len(rec) <= 0:
                    break

            print("\nlunghezza finale: ", len(data))
        except socket.timeout:
            print("timeout connessione")
            socket_client.send(timestamp)
            continue

        # from np.array of byte to PIL Image
        frame = generateNpArray(data)

        # from BGRA to BGR, remove opacity
        frame = frame[:, :, ::-1]

        # get aus (list of double)
        curr_aus = await detectAus(frame)

        if len(curr_aus[0]) > 0:
            # normalize aus
            aus_list = NormalizeData(curr_aus[0][0].tolist())

            print(aus_list)
            ausInByte = b''
            for aus in aus_list:
                ausInByte += fromIntToByte(aus)

            socket_client.send(timestamp + ausInByte)  # send to server
        else:
            socket_client.send(timestamp)

        await asyncio.sleep(0.01)


# Convert array of byte in np array, readable from py feat
def generateNpArray(frame):
    return np.frombuffer(frame, dtype='uint8').reshape((1080, 1920, 4), order='C')


# detect aus with py-feat
async def detectAus(frame):
    detected_face = detector.detect_faces(frame)
    detected_landmarks = detector.detect_landmarks(frame, detected_face)
    return detector.detect_aus(frame, detected_landmarks)


# normalize data in scale 0 to 100
def NormalizeData(data: list):
    return [int(val * 100) for val in data]


# Convert Double (8 byte) into float (4 byte)
def fromIntToByte(val):
    byte_length = (val.bit_length() + 7) // 8 or 1
    return val.to_bytes(byte_length, byteorder='little')


if __name__ == "__main__":
    socket_client.connect((IP, PORT))
    asyncio.run(mainLoop())
