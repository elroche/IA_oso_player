import numpy as np
from pynput import keyboard
from PIL import ImageGrab , Image
import time
from queue import Queue

fps  = 30
myshape = (1080,1920)
# frame_written = 0

use_flag = False
pausing_flag  = False

def on_release(key):
    global pausing_flag
    if key == keyboard.Key.esc or key.char == 'p': 
        pausing_flag = not pausing_flag
        print("Pipeline paused" if pausing_flag == False else "Pipeline started")

def pipeline_thread(que : Queue) -> None:
    T = time.time()
    print("Pipeline Thread entered , press P or Esc to start")
    with keyboard.Listener(on_press=None, on_release=on_release) as listener:
        while True:
            Y = time.time()
            if pausing_flag and (Y-T>1/fps):
                T = time.time()
                screenshot = np.array(ImageGrab.grab().convert('L'))
                que.put(screenshot)
            
