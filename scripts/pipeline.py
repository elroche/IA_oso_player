import numpy as np
from PIL import ImageGrab , Image
import time

fps  = 30
myshape = (1080,1920)
shared_array = np.memmap("../tmp/testarray", mode='w+', shape=myshape)
frame_written = 0
T = time.time()

while True:
    Y = time.time()
    if(Y-T>1/fps):
        T = time.time()
        screenshot = np.array(ImageGrab.grab().convert('L'))
        shared_array[:] = screenshot[:]
        frame_written += 1
        print(frame_written)
