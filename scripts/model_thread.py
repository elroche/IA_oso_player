import matplotlib.pyplot as plt

import threading
from queue import Queue

import torch
from torch import Tensor

from tesseract_model import process_image
from reinforcement_model import inference
from pipeline_thread import pipeline_thread

import pyautogui

# Create a queue for communication between threads
que = Queue()

# Thread function to continuously record received images.
def test_thread(que : Queue):
    print("Test Thread entered")
    try:
        while True:
            img = que.get()
            plt.imsave("tmp/test.jpg" , img , cmap = 'grey')
    except Exception as e:
        print(f"Error in save_image: {e}")
        return -1

# Get initial mouse coordinates and normalize them
x_in , y_in = pyautogui.position()
x_in = torch.Tensor([[x_in]])/1920
y_in = torch.Tensor([[y_in]])/1080
prev_action = torch.zeros((1,4))

# Create two threads
thread1 = threading.Thread(target=pipeline_thread, args=(que,))
thread2 = threading.Thread(target=inference,args=(que,x_in,y_in, prev_action,))

# Start the threads
thread1.start()
thread2.start()

# Wait for both threads to finish
thread1.join()
thread2.join()

print("Both threads have finished.")
