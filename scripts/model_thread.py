import numpy as np
import matplotlib.pyplot as plt

from PIL import ImageGrab, Image

import threading
from queue import Queue

from tesseract_model import tesseract_model
from pipeline_thread import pipeline_thread

que = Queue()

def test_thread(que : Queue):
    print("Test Thread entered")
    try:
        while True:
            print(que.qsize())
            # img = que.get()
            # plt.imsave("../tmp/test.jpg" , img , cmap = 'grey')
            # print("Image saved")
    except Exception as e:
        print(f"Error in save_image: {e}")
        return -1


# Create two threads
thread1 = threading.Thread(target=pipeline_thread, args=(que,))
thread2 = threading.Thread(target=test_thread, args=(que,))

# Start the threads
thread1.start()
thread2.start()

# Wait for both threads to finish
thread1.join()
thread2.join()

print("Both threads have finished.")
