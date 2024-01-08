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
