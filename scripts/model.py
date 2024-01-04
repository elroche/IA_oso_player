import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageGrab, Image
import time
from queue import Queue
import threading
from model_definition import process_image

que = Queue()


def get_pipeline():
    fps = 30
    read_shape = (1080, 1920)
    shared_array = np.memmap("../tmp/screenshot", mode='r', shape=read_shape)
    T = time.time()

    global que
    try:
        while True:
            Y = time.time()
            if (Y - T > 1 / fps):
                T = time.time()
                img = shared_array[:]
                que.put(img)
    except Exception as e:
        print(f"Error in get_pipeline: {e}")
        return -1


def save_image():
    global que
    try:
        while True:
            img = que.get()
            processed_image, _, _, score, precision = process_image(
                Image.fromarray(img))
            # Print the detected score and precision
            print("Detected Score:", score)
            print("Detected Precision:", precision)
    except Exception as e:
        print(f"Error in save_image: {e}")
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
