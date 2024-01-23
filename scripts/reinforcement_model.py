from image_preprocessing_model import ContourDetector
import torch
import torch.nn as nn
import pyautogui
from PIL import Image
from PIL import ImageGrab
from torchvision import transforms
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import os
import matplotlib.pyplot as plt

contour_model = ContourDetector()


class ReinforcementModel(nn.Module):
    def __init__(self, hidden_size):
        super(ReinforcementModel, self).__init__()

        # LSTM layer
        self.lstm = nn.Linear(8160 + 1 + 1 + 4, hidden_size)

        # Fully connected layers for unified predictions
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, 32)

        # Output layers
        self.fc_action = nn.Linear(32, 4)
        self.fc_x = nn.Linear(32, 1)
        self.fc_y = nn.Linear(32, 1)

    def forward(self, flattened_image, x_in, y_in, prev_action, a):
        # print("X IN 2", x_in)
        # print("Y IN 2", y_in)
        # print("Dimension prev_action", prev_action.size())
        # Reshape prev_action to match the other tensors
        prev_action = prev_action.view(1, -1)
        # print("Dimension prev_action", prev_action.size())

        # print(flattened_image.size())
        # print(x_in.size())
        # print(y_in.size())
        # print(prev_action.size())
        # Concatenate flattened image with x_in, y_in, and prev_action
        input_concatenated = torch.cat(
            (flattened_image, x_in, y_in, prev_action), dim=1)
        # print("Concatenated Input Size:", input_concatenated.size())

        # LSTM layer
        out = self.lstm(input_concatenated.view(
            input_concatenated.size(0), 1, -1))

        print(a.any() == out.any())
        a = out

        # Only take the output from the last time step
        # out = out[:, -1, :]
        # print("LSTM Output Size:", out.size())

        out = torch.relu(out)

        # Fully connected layers
        out = torch.relu(self.fc2(torch.relu(self.fc1(out))))
        out = torch.relu(self.fc4(torch.relu(self.fc3(out))))
        # print("Fully Connected Output Size:", out.size())

        # Action prediction
        action_out = torch.softmax(self.fc_action(out), dim=1)
        # print("Action Output Size:", action_out.size())

        print("X OUT :", self.fc_x(out))
        # print("REGARDE ICI", self.fc_y(out))
        # x_out prediction
        x_out = torch.abs(torch.tanh(self.fc_x(out)))
        # print("X Output Size:", x_out.size())
        print("tnah X OUT :", x_out)

        # y_out prediction
        y_out = torch.abs(torch.tanh(self.fc_y(out)))
        # print("Y Output Size:", y_out.size())
        # print("Y OUT:", y_out)

        # print("Sortie de la couche x_out:", x_out)
        # print("Sortie de la couche y_out:", y_out)

        return action_out, x_out, y_out, a


# Obtenir les dimensions de l'écran principal
screen_width, screen_height = pyautogui.size()
print("LARGEUR DE L ECRAN", screen_width)
print("LONGUEUR DE L ECRAN", screen_height)


# Fonction pour exécuter l'action et le déplacement
def execute_action_and_move(action, x_out, y_out):
    # Convertir la sortie du modèle en action réelle
    action_index = torch.argmax(action, dim=1).item()

    # Exécuter l'action correspondante
    if action_index == 0:  # Exemple : "clic"
        pyautogui.click()
    elif action_index == 1:  # Exemple : "clic_down"
        pyautogui.mouseDown()
    elif action_index == 2:  # Exemple : "clic_up"
        pyautogui.mouseUp()
    else:  # action_index == 3, "None"
        pass  # Aucune action

    # Déplacer la souris vers la position
    # print("X OUT", x_out)
    # print("Y OUT", y_out)
    target_x = int(x_out.item()*screen_width)
    target_y = int(y_out.item()*screen_height)
    pyautogui.moveTo(target_x, target_y)
    # print("TARGET X", target_x)
    # print("TARGET Y", target_y)


input_size = 270 * 480
hidden_size = 128
num_classes = 4  # clic, clic_down, clic_up, None
lag = 20
a = np.array([])

model = ReinforcementModel(hidden_size)


####################################### Tests #######################################

def test_recuperation_image(folder_path="images_test"):
    screenshots = []
    for i in range(19):
        image_path = os.path.join(folder_path, f"image_{i+1}.jpg")
        screenshot = Image.open(image_path).convert('L')
        screenshots.append(screenshot)
    return screenshots


"""
def test_recuperation_image():
    i = 0
    screenshots = []
    while (i < 20):
        screenshot = ImageGrab.grab().convert('L')
        screenshots.append(screenshot)
        i += 1
    return screenshots
"""


screenshots = test_recuperation_image()

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize([0.5, ], [0.5, ])])


# Initialisez avec la classe "None"
prev_action = torch.zeros((1, num_classes))
# print("prev_action : ", prev_action)
# print("taille prev_action : ", prev_action.size())

# initilaisation x_in et y_in
x_in = torch.randint(low=1, high=1920, size=(1, 1))/1920
y_in = torch.randint(low=1, high=1080, size=(1, 1))/1080
print("X in", x_in)
# print("Y IN", y_in)

for i, image_data in enumerate(screenshots):
    print("i : ", i)
    # Prétraitement avec le modèle de détection de contours
    image_data = transform(image_data)
    _, contours = contour_model(image_data)
    print(contours.detach().numpy().sum())
    # Maintenant, contours contient les résultats de la détection de contours
    # print("Dimensions de la première image après contour 1: ", contours.shape)

    action_out, x_out, y_out, a = model(
        contours, x_in, y_in, prev_action, a)

    # execute_action_and_move(action_out, x_out, y_out)

    # Test, Mise à jour pour la prochaine itération
    # x_in = torch.randint(low=1, high=1920, size=(1, 1))/1920
    # y_in = torch.randint(low=1, high=1080, size=(1, 1))/1080
    print("X in : ", x_in)
    # print("Y CHANGE : ", y_in)
    prev_action = action_out
