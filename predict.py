import torchvision.models as models
import torch.nn as nn
import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import manip

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
digits = manip.get_grade("cropped_grade.png")


model = models.resnet18(weights=None)

# Change first conv layer to accept 1 channel instead of 3
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

# Adjust final layer for 10 MNIST classes
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)

# Load the trained weights
model.load_state_dict(torch.load("mnist_resnet18.pth", map_location="mps"))

model.to(device)
model.eval()
transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ]
)

predictions = []

for digit in digits:
    img_t = manip.preprocess_for_resnet(digit).to(device)
    with torch.no_grad():
        output = model(img_t)
        pred_class = torch.argmax(output, dim=1).item()
        predictions.append(pred_class)

print(predictions)
