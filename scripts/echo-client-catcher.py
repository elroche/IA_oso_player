# echo-client.py

import socket
import pickle

HOST = "127.0.0.1"  # The server's hostname or IP address
PORT = 8888  # The port used by the server

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    print('Connexion Established')

    while True:
        print('in true')
        data = s.recv(4096)
        print('Data Catch.')
        data_arr = pickle.loads(data)
        print('Received', repr(data_arr))
