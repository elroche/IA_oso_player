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

# Fonction de la détection du score et de la précision
def detect_numbers(image):
    # Charger un modèle Faster R-CNN pré-entraîné
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = fasterrcnn_resnet50_fpn(pretrained=True)
        model.eval()
    warnings.resetwarnings()

    # Récupérer les dimensions originales de l'image
    width, height = image.size

    # Définir les coordonnées de la région à traiter
    region_left = int(0.80 * width)
    region_top = 0
    region_right = width
    region_bottom = int(0.15 * height)

    # Extraire la région à traiter
    region = image.crop((region_left, region_top, region_right, region_bottom))
    # plt.imshow(region); plt.show()

    # Appliquer les transformations nécessaires à la région
    region_tensor = functional.to_tensor(region)
    region_tensor = region_tensor.unsqueeze(
        0)  # Ajouter une dimension pour le lot

    # Effectuer l'inférence sur la région
    with torch.no_grad():
        prediction = model(region_tensor)

    # Utiliser Tesseract OCR pour reconnaître le texte dans la région
    result = pytesseract.image_to_string(region)
    # Filtrer les caractères pour ne conserver que les nombres, et le retour à a ligne pour séparer les 2 résultats
    filtered_result = re.sub(r'[^0-9\n,.]', '', result)
    # Retourner également le résultat filtré pour une utilisation ultérieure
    return prediction, filtered_result


# Fonction pour extraire le score et la précision à partir du filtered_result
def extraction_score_precision(filtered_result):
    # Vérifier si filtered_result est une chaîne vide ou ne contient que des espaces
    if not filtered_result or filtered_result.isspace() or filtered_result.isalpha():
        print("Aucun nombre détecté.")
        return None, None

    # Extraire le score (le score étant sur la première ligne)
    lines = filtered_result.split('\n')

    # Vérifier si lines a au moins une ligne
    if not lines:
        print("Aucune ligne trouvée.")
        return None, None

    # Accéder à la première ligne
    first_line = lines[0].strip().split()

    # Vérifier si la première ligne a suffisamment d'éléments
    if not first_line:
        print("Aucun élément trouvé dans la première ligne.")
        return None, None

    score_str = first_line[-1]
    score = int(score_str)

    # Extraire la précision (la précision étant sur la deuxième ligne)
    precision_str = lines[2].strip().replace('%', '')
    # Extraire la précision (la précision étant sur la deuxième ligne)
    precision_str = lines[2].strip().replace('%', '')
    # Remplacer la virgule par un point
    precision_str = precision_str.replace(',', '.')
    precision = float(precision_str)

    return score, precision


# Fonction générale de détection du score et de la précision
def process_image(image):
    # Detection des nombres sur l'image
    T = time.time()
    _, filtered_result = detect_numbers(image)

    if filtered_result is None:
        print("Aucun nombre détecté. Traitement arrêté.")
        return None , None
    # Extraire le score et la précision
    score, precision = extraction_score_precision(filtered_result)
    print(time.time() - T)
    return score, precision

# Fonction de comparaison du score et de la précision de 2 images
def scores_precision_difference(filtered_result_precedente, filtered_result_actuelle):
    # Extraire le score et la précision des prédictions
    previous_score, previous_precision = extraction_score_precision(
        filtered_result_precedente)
    current_score, current_precision = extraction_score_precision(
        filtered_result_actuelle)

    if previous_score is None or current_score is None or previous_precision is None or current_precision is None:
        print("Aucun nombre détecté. Impossible de calculer la différence.")
        return None, None

    # Calculer la différence du score et de la précision
    score_difference = current_score - previous_score
    precision_difference = current_precision - previous_precision

    return score_difference, precision_difference

def tesseract_model(que : Queue):
    try:
        while True:
            img = que.get()
            score, precision = process_image(img)
            # Print the detected score and precision
            print("Detected Score:", score)
            print("Detected Precision:", precision)
    except Exception as e:
        print(f"Error in tesseract_model: {e}")
        return -1
    
# img = Image.open("../tmp/test.jpg")
# process_image(img)
# que = Queue()
# que.put(img)
# tesseract_model(que)
