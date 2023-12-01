import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageGrab , Image
import time

fps =30
read_shape = (1080,1920)
shared_array = np.memmap("../tmp/screenshot", mode='r', shape=read_shape)
T = time.time()

while True:
    Y = time.time()
    if(Y-T>1/fps):
        T = time.time()
    # shared_array will behave as a numpy ndarray
        img = shared_array[:]
        plt.imsave('../tmp/test.jpg',img,cmap='Greys_r')