import torch

from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional

import pytesseract

import matplotlib.pyplot as plt

import re
import warnings
import time
from queue import Queue

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Score and accuracy detection function
def detect_numbers(image):
    width, height = image.size

    # Definition of the coordinates of the region to be treated
    region_left = int(0.80 * width)
    region_top = 0
    region_right = width
    region_bottom = int(0.15 * height)

    region = image.crop((region_left, region_top, region_right, region_bottom))

    # plt.imsave("tmp/test_region.jpg", region, cmap='gray')

    # Using Tesseract OCR to recognise text in the region
    result = pytesseract.image_to_string(region)
    # Character filtering
    filtered_result = re.sub(r'[^0-9\n,.]', '', result)

    return filtered_result


# Function for extracting score and precision from the filtered result
def extraction_score_precision(filtered_result):
    # Check that filtered_result is not an empty string or contains only spaces
    if not filtered_result or filtered_result.isspace() or filtered_result.isalpha():
        print("Aucun nombre détecté.")
        return None

    # Extraction of the score (the score is on the first line)
    lines = filtered_result.split('\n')

    # Check that lines are not invalid
    if not lines:
        print("Aucune ligne trouvée.")
        return None
    elif lines[0] == '':
        print("Aucun nombre détecté.")
        return None

    return float(lines[0])


# General score and precision detection function
def process_image(image):
    # Detecting numbers in the image
    filtered_result = detect_numbers(image)
    # Score extraction and accuracy
    return extraction_score_precision(filtered_result)


# Function to load a pre-trained Faster R-CNN model and continuously process images from a queue.
def tesseract_model(que: Queue):
    # Try to continuously process images from the queue
    try:
        while True:
            T = time.time()
            img = que.get()
            precision  = process_image(img)
            print('Precision : ', precision)
            # print("Time taken :" ,time.time() - T)
    except Exception as e:
        print(f"Error in tesseract_model: {e}")
        return -1
