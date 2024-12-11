from freenect2 import Device
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
                while True:
                    conn, addr = skt.accept()
                    with conn:
                        print(f"Request accepted")
                        type_, frame = device.get_next_frame()
                        # print(frame.to_array().tobytes())
                        conn.sendall(frame.data)
                        print(f"Frame sent")

            except KeyboardInterrupt:
                print("Close from user")

if __name__ == "__main__":
    start_server()