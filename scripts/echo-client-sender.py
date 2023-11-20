# echo-client.py

import socket
import pickle

HOST = "127.0.0.1"  # The server's hostname or IP address
PORT = 8888  # The port used by the server

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))

    arr = ([1,2,3,4,5,6],[1,2,3,4,5,6])
    while True:
        data_string = pickle.dumps(arr)
        s.sendall(data_string)
        print('Array send')

        # data = s.recv(4096)
        # data_arr = pickle.loads(data)
        # # s.close() 
        # print('Received', repr(data_arr))
