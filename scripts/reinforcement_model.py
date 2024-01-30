from image_preprocessing_model import ContourDetector
import torch
import torch.nn as nn
import pyautogui
from torchvision import transforms
import queue

contour_model = ContourDetector()

# ReinforcementModel class definition for a neural network model
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

    def forward(self, flattened_image, x_in, y_in, prev_action):

        # Reshape prev_action to match the other tensors
        prev_action = prev_action.view(1, -1)
        # Concatenate flattened image with x_in, y_in, and prev_action
        input_concatenated = torch.cat(
            (flattened_image, x_in, y_in, prev_action), dim=1)

        # LSTM layer
        out = self.lstm(input_concatenated.view(
            input_concatenated.size(0), 1, -1))

        out = torch.relu(out)

        # Fully connected layers
        out = torch.relu(self.fc2(torch.relu(self.fc1(out))))
        out = torch.relu(self.fc4(torch.relu(self.fc3(out))))

        # Action prediction
        action_out = torch.softmax(self.fc_action(out), dim=1) # peut etre 1 1 1 1 a cause du softmax
        # x_out prediction
        x_out = torch.abs(torch.tanh(self.fc_x(out)))
        # y_out prediction
        y_out = torch.abs(torch.tanh(self.fc_y(out)))
        return action_out, x_out, y_out


# Get main screen dimensions
screen_width, screen_height = pyautogui.size()

input_size = 270 * 480
hidden_size = 128
num_classes = 4  # clic, clic_down, clic_up, None
lag = 20

# Model initialization
model = ReinforcementModel(hidden_size)

# To convert PIL images to PyTorch tensors and pixel values to normalized values.
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5, ], [0.5, ])])

# Function for continuous inference using the provided queue, current x and y coordinates, and the previous action list.
def inference(que : queue.Queue , x_in : float , y_in : float , prev_action : list) -> None:
    while True:
        # Get screenshot from the queue
        screenshot = que.get()
        # Initialize previous action as zeros
        prev_action = torch.zeros((1, num_classes))
        # Preprocessing with the edge detection model
        image_data = transform(screenshot)
        _, contours = contour_model(image_data)
        # Run inference using the main model
        action_out, x_out, y_out = model(
            contours, x_in, y_in, prev_action)
        execute_action_and_move(action_out, x_out, y_out)


# Function to execute action and move
def execute_action_and_move(action, x_out, y_out):
    # Convert model output to real actionp
    action_index = torch.argmax(action, dim=2).item()

    # Execute the corresponding action
    if action_index == 0:  # Exemple : "clic"
        pyautogui.click()
    elif action_index == 1:  # Exemple : "clic_down"
        pyautogui.mouseDown()
    elif action_index == 2:  # Exemple : "clic_up"
        pyautogui.mouseUp()
    else:  # action_index == 3, "None"
        pass  # None action

    # Move mouse to position
    target_x = int(x_out.item()*screen_width)
    target_y = int(y_out.item()*screen_height)
    pyautogui.moveTo(target_x, target_y)