import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# --- 1. Device ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# --- 2. Load trained ResNet18 ---
model = models.resnet18(weights=None)
model.conv1 = nn.Conv2d(
    1, 64, kernel_size=7, stride=2, padding=3, bias=False
)  # 1-channel input
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)
model.load_state_dict(torch.load("mnist_resnet18.pth", map_location=device))
model.to(device)
model.eval()


# --- 3. Extract red digits ---
def get_grade(img_path):
    img = cv2.imread(img_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Red hue ranges
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    bounding_boxes_sorted = sorted(bounding_boxes, key=lambda b: b[0])

    digits = []
    for x, y, w, h in bounding_boxes_sorted:
        digit_img = red_mask[y : y + h, x : x + w]
        digits.append(digit_img)

    return digits


# --- 4. Preprocess digits ---
def preprocess_for_resnet(roi, target_size=224):
    # Make sure digit is white on black
    roi = cv2.bitwise_not(roi)
    h, w = roi.shape

    # Resize and pad to square
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)

    delta_w = target_size - new_w
    delta_h = target_size - new_h
    top, bottom = delta_h // 2, delta_h - delta_h // 2
    left, right = delta_w // 2, delta_w - delta_w // 2
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0
    )

    # Normalize to [-1, 1]
    tensor = torch.tensor(padded, dtype=torch.float32) / 255.0
    tensor = (tensor - 0.5) / 0.5

    # Add batch and channel dimensions
    tensor = tensor.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    return tensor


# --- 5. Predict digits ---
def predict_digits(img_path):
    digits = get_grade(img_path)
    predictions = []

    for i, digit in enumerate(digits):
        img_t = preprocess_for_resnet(digit).to(device)
        with torch.no_grad():
            output = model(img_t)
            pred_class = torch.argmax(output, dim=1).item()
            predictions.append(pred_class)
        print(f"Digit {i}: predicted as {pred_class}")

    return predictions


# --- 6. Run pipeline ---
if __name__ == "__main__":
    img_path = "cropped_grade.png"  # your red handwritten grade image
    preds = predict_digits(img_path)
    print("All predicted digits:", preds)
