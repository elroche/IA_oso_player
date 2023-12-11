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

torch.cuda.is_available()

# Classe de détecion de contour
# (utilisation de conv2d (CNN) sur les images)
class ContourDetector(nn.Module):
    def __init__(self):

        super(ContourDetector, self).__init__()

        # Couches de convolution
        self.Conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.Conv2 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1) # : division de la taille de la matrice de contours par 2, je trouve ça bien comme résultat

        # Couches de pooling
        self.Pool = nn.MaxPool2d(kernel_size=1, stride=2, padding=0) # division de la taille de la matrice de contours par 4(stride ici a 2 et a la 2e couche de conv stride à 1)
        self.Flat=nn.Flatten()


    def forward(self, x):
        # Passe avant les couches de convolution
        x = self.Pool(F.relu(self.Conv1(x)))
        z = self.Pool(F.relu(self.Conv2(x)))

        y = self.Flat(z)
        print("Dimensions avec shape:", y.shape)

        return z,y


# Fonction de la détection du score et de la précision
def detect_numbers(image):
    # Charger un modèle Faster R-CNN pré-entraîné
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()

    # Récupérer les dimensions originales de l'image
    width, height = image.size

    # Définir les coordonnées de la région à traiter (1/4 de la longueur et 1/4 de la hauteur)
    region_left = int(0.80 * width)
    region_top = 0
    region_right = width
    region_bottom = int(0.15 * height)

    # Extraire la région à traiter
    region = image.crop((region_left, region_top, region_right, region_bottom))

    # Appliquer les transformations nécessaires à la région
    region_tensor = functional.to_tensor(region)
    region_tensor = region_tensor.unsqueeze(0)  # Ajouter une dimension pour le lot

    # Effectuer l'inférence sur la région
    with torch.no_grad():
        prediction = model(region_tensor)

    # Convertir la région en niveaux de gris
    region_gray = region.convert('L')

    # Utiliser Tesseract OCR pour reconnaître le texte dans la région
    result = pytesseract.image_to_string(region_gray)

    # Filtrer les caractères pour ne conserver que les nombres, le caractère "%", et le retour à a ligne pour séparer les 2 résultats
    filtered_result = re.sub(r'[^0-9%\n,.]', '', result)

    return prediction, region, filtered_result  # Retourner également le résultat filtré pour une utilisation ultérieure

# Fonction pour extraire le score et la précision à partir du filtered_result
def extraire_score_precision(filtered_result):
    # Extraire le score (le score étant sur la première ligne)
    lignes = filtered_result.split('\n')
    score_str = lignes[0].strip().split()[-1]
    score = int(score_str)

    # Extraire la précision (la précision étant sur la deuxième ligne)
    precision_str = lignes[2].strip().replace('%', '')
    precision_str = precision_str.replace(',', '.')  # Remplacer la virgule par un point
    precision = float(precision_str)

    return score, precision


# Fonction générale de détection du score et de la précision
def process_image(image):
    # Detection des nombres sur l'image
    _, region, filtered_result = detect_numbers(image)
    # Extraire le score et la précision
    score, precision = extraire_score_precision(filtered_result)
    return image, region, filtered_result, score, precision


# Fonction de comparaison du score et de la précision de 2 images
def difference_scores_precisions(filtered_result_precedente, filtered_result_actuelle):
    # Extraire le score et la précision des prédictions
    score_precedent, precision_precedente = extraire_score_precision(filtered_result_precedente)
    score_actuel, precision_actuelle = extraire_score_precision(filtered_result_actuelle)

    # Calculer la différence du score et de la précision
    difference_score = score_actuel - score_precedent
    difference_precision = precision_actuelle - precision_precedente

    return difference_score, difference_precision


# Initialisation du modèle, à mettre dans fonction
contour_model = ContourDetector()
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5, ], [0.5, ])])
print(contour_model)


##################### Test des fonctions #####################

def test_recuperation_image() :
    i = 0
    screenshots = []
    while (i<200) :
        screenshot = ImageGrab.grab().convert('L')
        screenshots.append(screenshot)
        i += 1
    return screenshots

# Récupération des screenshots, conversion en tableau NumPy, puis en objet PIL.Image (sera fait dans le traitement)
screenshots = test_recuperation_image()
image_precedente = screenshots[-8]
image_actuelle = screenshots[-1]
image_precedente_np = np.array(image_precedente)
image_actuelle_np = np.array(image_actuelle)
pil_image_precedente = Image.fromarray(image_precedente_np)
pil_image_actuelle = Image.fromarray(image_actuelle_np)

# Appeler la fonction globale de détection
pil_image_precedente, region_precedente, filtered_result_precedente, score_precedent, precision_precedente = process_image(pil_image_precedente)
pil_image_actuelle, region_actuelle, filtered_result_actuelle, score_actuel, precision_actuelle = process_image(pil_image_actuelle)

# Extraction du score et de la précision des prédictions faites sur les images (pas obligatoire car fait dans la fonction extraire_score_precision)
print("Score de l'image actuelle :", score_actuel)
print("Precision de l'image actuelle :", precision_actuelle)
print("Score de l'image précédente :", score_precedent)
print("Precision de l'image précédente :", precision_precedente)

# Test de la fonction difference_scores_precisions
difference_score, difference_precision = difference_scores_precisions(filtered_result_precedente, filtered_result_actuelle)

# Afficher les résultats du test, cad de la différence du score et de la précision entre les deux images
print("Différence du score :", difference_score)
print("Différence de la précision :", difference_precision)