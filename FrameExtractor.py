from freenect2 import Device, FrameType
import socket


IP = '127.0.0.1'
PORT = 8053

# Opzioni:
# 1. script che cattura frame, appena riceve una richiesta lo fornisce.
# 2. cript che fornisce delle


# implementazione opzione 1.

def start_server():
    device = Device()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as skt:
        with device.running():
            skt.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            skt.bind((IP, PORT))
            skt.listen()
            print(f"Server listen on {IP}:{PORT}...")

            try:
                conn, addr = skt.accept()
                while True:
                    with conn:
                        print(f"Request accepted")
                        for i, (type_, frame) in enumerate(device):
                            if FrameType.Color is type_:
                                conn.sendall(frame.to_array()[:720, :1280, 0:3].tobytes())
                                print(f"Frame sent")

            except KeyboardInterrupt:
                print("Close from user")

if __name__ == "__main__":
    start_server()