from freenect2 import Device, FrameType
import socket

IP = '127.0.0.1'
PORT = 8052
socket_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Opzioni:
# 1. script che cattura frame, appena riceve una richiesta lo fornisce.
# 2. cript che fornisce delle


# implementazione opzione 1.

device = Device()

device.start()
try:
    while True:
        type_, frame = device.get_next_frame()
        socket_client.sendall(frame.tobytes())

finally:
    device.stop()

