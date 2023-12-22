import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageGrab , Image
import time
from queue import Queue  
import threading


que = Queue() 

def get_pipeline():
    fps =30
    read_shape = (1080,1920)
    shared_array = np.memmap("../tmp/screenshot", mode='r', shape=read_shape)
    T = time.time()

    global que
    try:
        while True:
            Y = time.time()
            if(Y-T>1/fps):
                T = time.time()
            # shared_array will behave as a numpy ndarray
                img = shared_array[:]
                que.put(img)
    except:
        return -1

def save_image():
    global que
    try:
        while True:
            img = que.get()
            #this is where u have image cropping and number detection
            # the img object is a numpy array containing the full screenshot size 1080,1920
    except:
        return -1
# Create two threads
thread1 = threading.Thread(target=get_pipeline)
thread2 = threading.Thread(target=save_image)

# Start the threads
thread1.start()
thread2.start()

# Wait for both threads to finish
thread1.join()
thread2.join()

print("Both threads have finished.")
