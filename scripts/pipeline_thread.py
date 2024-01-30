from pynput import keyboard
from PIL import ImageGrab
import time
from queue import Queue

# Initialization of frames per second and screen resolution.
fps  = 30
myshape = (1080,1920)

# Flags to control the pipeline
use_flag = False
pausing_flag  = False

# Callback function triggered when the "p" key or "escape" key is released.
def on_release(key):
    global pausing_flag
    try:
        if key == keyboard.Key.esc or key.char == 'p': 
            pausing_flag = not pausing_flag
            print("Pipeline paused" if pausing_flag == False else "Pipeline started")
    except:
        pass

# Thread function for continuously capturing screenshots and putting them into the queue.
def pipeline_thread(que : Queue) -> None:
    T = time.time()
    print("Pipeline Thread entered , press P or Esc to start")
    
    # Start keyboard listener to detect key presses and releases
    with keyboard.Listener(on_press=None, on_release=on_release) as listener:
        while True:
            Y = time.time()
            # Check if pausing_flag is True and time elapsed is greater than 1/fps
            if pausing_flag and (Y-T>1/fps):
                T = time.time()
                # Capture screenshot and convert to grayscale
                screenshot = ImageGrab.grab(bbox=(0,0,1920,1080)).convert('L')
                # Put the screenshot into the queue for further processing
                que.put(screenshot)
            