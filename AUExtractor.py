import struct
import asyncio
import socket
from PIL import Image
from feat import Detector

# SETUP
# detector used to acquire faces, landmarks and AUs
detector = Detector(face_model='faceboxes', landmark_model='mobilefacenet', au_model='xgb')

# socket setup
IP = ''
PORT = 0
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

        # get all data from socket
        for i in range(0, 8100):
            cd = socket_client.recv(4096)
            data += cd
            print("curr pkt len ", len(cd), " TOT LEN ", len(data))

        print(data[0:15])
        print("lunghezza finale: ", len(data))

        # Unpack recived data
        timestamp, frame = unpackData(data)

        # from np.array of byte to PIL Image
        frame = generateImage(frame)

        # from BGRA to BGR, remove opacity
        frame = frame[:, :, ::-1]

        # get aus (list of double)
        curr_aus = await detectAus(frame)

        print(socket_client.send(timestamp))

        if len(curr_aus[0]) > 0:
            # normalize aus
            aus_list = NormalizeData(curr_aus[0][0].tolist())

            # convert AUs into
            for au in range(len(aus_list)):
                aus_list[au] = fromDoubleToFloat(aus_list[au])

            socket_client.send(timestamp + b''.join(struct.pack('d', num) for num in aus_list))  # send to server
        await asyncio.sleep(0.01)


# Convert array of byte in PIL image, readable from py feat
def generateImage(frame):
    return Image.frombytes('RGBA', (1920, 1080), frame, 'raw', 'BGRA')


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


# normalize data in scale 0 to 100
def NormalizeData(data: list):
    normalized_data = [int(val * 1000) / 10 for val in data]
    return normalized_data


# Convert Double (8 byte) into float (4 byte)
def fromDoubleToFloat(val):
    return struct.pack('>f', struct.unpack('>d', val)[0])


if __name__ == "__main__":
    socket_client.connect((IP, PORT))
    asyncio.run(mainLoop())
