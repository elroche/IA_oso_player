import torch

from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional

import pytesseract

import matplotlib.pyplot as plt
import numpy as np


import re
import warnings
import time
from PIL import Image
from queue import Queue

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Score and accuracy detection function


def detect_numbers(image, model):
    width, height = image.size

    # Definition of the coordinates of the region to be treated
    region_left = int(0.80 * width)
    region_top = 0
    region_right = width
    region_bottom = int(0.15 * height)

    region = image.crop((region_left, region_top, region_right, region_bottom))

    plt.imsave("tmp/test_region.jpg", region, cmap='gray')

    # Application of the necessary transformations to the region
    region_tensor = functional.to_tensor(region)
    region_tensor = region_tensor.unsqueeze(0)

    # Using Tesseract OCR to recognise text in the region
    result = pytesseract.image_to_string(region)
    # Character filtering
    filtered_result = re.sub(r'[^0-9\n,.]', '', result)

    return filtered_result


# Function for extracting score and precision from filtered_result
def extraction_score_precision(filtered_result):
    # Check that filtered_result is not an empty string or contains only spaces
    if not filtered_result or filtered_result.isspace() or filtered_result.isalpha():
        print("Aucun nombre détecté.")
        return None, None

    # Extraction of the score (the score is on the first line)
    lines = filtered_result.split('\n')
    print(lines)

    # Check that lines are not invalid
    if not lines:
        print("Aucune ligne trouvée.")
        return None, None

    first_line = lines[0].replace(',', '.')

    # Check that the first line has enough elements
    if len(lines) < 2:
        print("Aucun élément trouvé dans la première ligne.")
        return None, None

    score_str = first_line[-1]
    score = int(score_str)

    ''' # Extraire la précision (la précision étant sur la deuxième ligne)
    precision_str = lines[1].strip().replace('%', '')
    # Remplacer la virgule par un point
    precision_str = precision_str.replace(',', '.')
    precision = float(precision_str)'''
    precision = 0

    return score, precision


# General score and precision detection function
def process_image(image, model):
    # Detecting numbers in the image
    filtered_result = detect_numbers(image, model)

    if filtered_result is None:
        print("Aucun nombre détecté. Traitement arrêté.")
        return None, None
    # Score extraction and accuracy
    score, precision = extraction_score_precision(filtered_result)
    return score, precision

# Compares the score and accuracy of 2 images


def scores_precision_difference(filtered_result_precedente, filtered_result_actuelle):
    # Extraction of score and prediction accuracy
    previous_score, previous_precision = extraction_score_precision(
        filtered_result_precedente)
    current_score, current_precision = extraction_score_precision(
        filtered_result_actuelle)

    if previous_score is None or current_score is None or previous_precision is None or current_precision is None:
        print("Aucun nombre détecté. Impossible de calculer la différence.")
        return None, None

    # Calculation of score difference and precision
    score_difference = current_score - previous_score
    precision_difference = current_precision - previous_precision

    return score_difference, precision_difference


def tesseract_model(que: Queue):
    # Loading a pre-trained Faster R-CNN model
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = fasterrcnn_resnet50_fpn(pretrained=True)
        model.eval()
    warnings.resetwarnings()

    try:
        while True:
            T = time.time()
            img = que.get()
            img.save("tmp/test.jpg")
            # detect_numbers(img,model)
            #  score, precision = process_image(img,model)
            # Print the detected score and precision
            print("Queue lenght : ", que.qsize())
            # print("Detected Score:", score)
            # print("Detected Precision:", precision)
            print("Time taken :", time.time() - T)
    except Exception as e:
        print(f"Error in tesseract_model: {e}")
        return -1

# img = Image.open("../tmp/test.jpg")
# process_image(img)
# que = Queue()
# que.put(img)
# tesseract_model(que)
