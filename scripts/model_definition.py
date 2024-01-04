from PIL import ImageGrab
from PIL import Image
import cv2 as cv

import numpy as np
import matplotlib.pyplot as plt

import torch

from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional
from torchvision import transforms

import torch.nn.functional as F
import torch.nn as nn

import pytesseract

import time
import re
import os

import warnings

torch.cuda.is_available()

# Classe de détecion de contour
# (utilisation de conv2d (CNN) sur les images)


class ContourDetector(nn.Module):
    def __init__(self):

        super(ContourDetector, self).__init__()

        # Couches de convolution
        self.Conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        # : division de la taille de la matrice de contours par 2, je trouve ça bien comme résultat
        self.Conv2 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)

        # Couches de pooling
        # division de la taille de la matrice de contours par 4(stride ici a 2 et a la 2e couche de conv stride à 1)
        self.Pool = nn.MaxPool2d(kernel_size=1, stride=2, padding=0)
        self.Flat = nn.Flatten()

    def forward(self, x):
        x = self.Pool(F.relu(self.Conv1(x)))
        z = self.Pool(F.relu(self.Conv2(x)))
        y = self.Flat(z)

        return z, y


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

    # Appliquer les transformations nécessaires à la région
    region_tensor = functional.to_tensor(region)
    region_tensor = region_tensor.unsqueeze(
        0)  # Ajouter une dimension pour le lot

    # Effectuer l'inférence sur la région
    with torch.no_grad():
        prediction = model(region_tensor)

    # Convertir la région en niveaux de gris
    region_gray = region.convert('L')

    # Utiliser Tesseract OCR pour reconnaître le texte dans la région
    result = pytesseract.image_to_string(region_gray)

    # Filtrer les caractères pour ne conserver que les nombres, le caractère "%", et le retour à a ligne pour séparer les 2 résultats
    filtered_result = re.sub(r'[^0-9%\n,.]', '', result)

    # Retourner également le résultat filtré pour une utilisation ultérieure
    return prediction, region, filtered_result


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
    # Remplacer la virgule par un point
    precision_str = precision_str.replace(',', '.')
    precision = float(precision_str)

    return score, precision


# Fonction générale de détection du score et de la précision
def process_image(image):
    # Detection des nombres sur l'image
    _, region, filtered_result = detect_numbers(image)

    if filtered_result is None:
        print("Aucun nombre détecté. Traitement arrêté.")
        return None
    # Extraire le score et la précision
    score, precision = extraction_score_precision(filtered_result)
    return image, region, filtered_result, score, precision


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


# Initialisation du modèle, à mettre dans fonction
contour_model = ContourDetector()
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize([0.5, ], [0.5, ])])
# print(contour_model)


##################### Test des fonctions #####################

"""def test_recuperation_image():
    i = 0
    screenshots = []
    while (i < 200):
        screenshot = ImageGrab.grab().convert('L')
        screenshots.append(screenshot)
        i += 1
    return screenshots


# Récupération des screenshots, conversion en tableau NumPy, puis en objet PIL.Image (sera fait dans le traitement)
screenshots = test_recuperation_image()
previous_image = screenshots[-8]
current_image = screenshots[-1]
previous_image_np = np.array(previous_image)
current_image_np = np.array(current_image)
pil_previous_image = Image.fromarray(previous_image_np)
pil_current_image = Image.fromarray(current_image_np)

# Appeler la fonction globale de détection
pil_previous_image, region_precedente, filtered_result_precedente, previous_score, previous_precision = process_image(
    pil_previous_image)
pil_current_image, region_actuelle, filtered_result_actuelle, current_score, current_precision = process_image(
    pil_current_image)

# Extraction du score et de la précision des prédictions faites sur les images (pas obligatoire car fait dans la fonction extraction_score_precision)
print("Score de l'image actuelle :", current_score)
print("Precision de l'image actuelle :", current_precision)
print("Score de l'image précédente :", previous_score)
print("Precision de l'image précédente :", previous_precision)

# Test de la fonction scores_precision_difference
score_difference, precision_difference = scores_precision_difference(
    filtered_result_precedente, filtered_result_actuelle)

# Afficher les résultats du test, cad de la différence du score et de la précision entre les deux images
print("Différence du score :", score_difference)
print("Différence de la précision :", precision_difference)"""
