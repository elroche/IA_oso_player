import numpy as np
from pynput import keyboard
from PIL import ImageGrab , Image
import time
from queue import Queue

fps  = 30
myshape = (1080,1920)
# frame_written = 0

use_flag = False
pausing_flag  = True


def on_press(key):
    global use_flag
    try:
        if(key.char == 'p'):
            use_flag = not use_flag
            print("Pipeline no in use" if use_flag == False else "Pipeline used")
    except AttributeError:
        pass

def on_release(key):
    global pausing_flag
    if key == keyboard.Key.esc:
        pausing_flag = not pausing_flag
        print("Pipeline paused" if pausing_flag == False else "Pipeline started")

def pipeline_thread(que : Queue) -> None:
    T = time.time()
    print("Pipeline Thread entered")
    # Configurer les gestionnaires d'événements
    # with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    #     listener.join()
        # while use_flag:
        #     if pausing_flag:
    while True:
        Y = time.time()
        if(Y-T>1/fps):
            T = time.time()
            screenshot = np.array(ImageGrab.grab().convert('L'))
            # print("Screenshot done")
            que.put(screenshot)
            # print("Queue extended")
