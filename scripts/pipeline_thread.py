import numpy as np
from pynput import keyboard
from PIL import ImageGrab , Image
import time

fps  = 30
myshape = (1080,1920)
shared_array = np.memmap("../tmp/screenshot", mode='w+', shape=myshape)
frame_written = 0
T = time.time()

def on_press(key):
    try:
        print(f"Touche {key.char} pressée")
    except AttributeError:
        print(f"Touche spéciale {key} pressée")

def on_release(key):
    if key == keyboard.Key.esc:
        # Pour arrêter l'écoute lorsque la touche Échap est pressée
        return False

# Configurer les gestionnaires d'événements
with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()
while True:
    Y = time.time()
    if(Y-T>1/fps):
        T = time.time()
        screenshot = np.array(ImageGrab.grab().convert('L'))
        shared_array[:] = screenshot[:1080,:1920]
