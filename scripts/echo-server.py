# echo-server.py

import socket 
import pickle

HOST = "127.0.0.1"  # Standard loopback interface address (localhost)
PORT = 8888  # Port to listen on (non-privileged ports are > 1023)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 2)
    s.bind((HOST, PORT))

    s.listen(1)
    conn, addr = s.accept()

    print('Connected by', addr)

    while True:

        data = conn.recv(4096)

        if data:
            print('send data')
            conn.sendall(data)

    conn.close()
