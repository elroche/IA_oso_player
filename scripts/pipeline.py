import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import ImageGrab , Image
import time
import sys

fps  = 30

def numpy_to_bytes(arr: bytearray) -> str:
    arr_dtype = bytearray(str(arr.dtype), 'utf-8')
    arr_shape = bytearray(','.join([str(a) for a in arr.shape]), 'utf-8')
    sep = bytearray('|', 'utf-8')
    arr_bytes = arr.ravel().tobytes()
    to_return = arr_dtype + sep + arr_shape + sep + arr_bytes
    return to_return

T = time.time()
while True:
    Y = time.time()
    if(Y-T>1/fps):
        T = time.time()
        screenshot = np.array(ImageGrab.grab())
        print(numpy_to_bytes(screenshot))
        sys.stdout.flush()
