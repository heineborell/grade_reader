import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import cv2
import matplotlib.pyplot as plt

# Set device (prioritize Mac GPU, then CUDA, then CPU)
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"Using device: {device} (Mac GPU)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using device: {device} (NVIDIA GPU)")
else:
    device = torch.device("cpu")
    print(f"Using device: {device} (CPU)")

print(f"PyTorch version: {torch.__version__}")


# 1. Define the CNN Model
class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout1(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout1(x)
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


# 2. Load and preprocess MNIST data
print("\nLoading MNIST dataset...")
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# 3. Initialize model, loss, and optimizer
model = MNISTNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("\nModel architecture:")
print(model)


# 4. Training function
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        if batch_idx % 100 == 0:
            print(f"  Batch {batch_idx}/{len(loader)}, Loss: {loss.item():.4f}")

    accuracy = 100.0 * correct / total
    avg_loss = running_loss / len(loader)
    return avg_loss, accuracy


# 5. Evaluation function
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    accuracy = 100.0 * correct / total
    avg_loss = running_loss / len(loader)
    return avg_loss, accuracy


# 6. Train the model
print("\nTraining model...")
num_epochs = 5

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    train_loss, train_acc = train_epoch(
        model, train_loader, criterion, optimizer, device
    )
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

# 7. Save the model
torch.save(model.state_dict(), "mnist_model.pth")
print("\nModel saved as 'mnist_model.pth'")


# 8. Functions to extract and preprocess red pencil digits
def get_grade(img_path, display=False):
    """Extract red pencil digits from image."""
    img = cv2.imread(img_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Red hue range #1 (0-10)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    # Red hue range #2 (170-180)
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    # Combine both masks
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Assume `contours` contains all detected digit contours
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    # Sort bounding boxes by x (left to right)
    bounding_boxes_sorted = sorted(bounding_boxes, key=lambda b: b[0])  # b[0] is x
    # Extract digit images in left-to-right order
    digits = []
    for x, y, w, h in bounding_boxes_sorted:
        digit_img = red_mask[y : y + h, x : x + w]
        digits.append(digit_img)
    if display:
        for i, digit in enumerate(digits):
            cv2.imshow(f"Digit {i}", digit)
            cv2.waitKey(0)
            cv2.destroyWindow(f"Digit {i}")
    cv2.destroyAllWindows()
    return digits


def preprocess_char(roi, size=28):
    """Preprocess a single digit image for MNIST model.

    Converts black digit on white background -> white digit on black background (like MNIST)
    """
    # roi from get_grade is white digit on black background (from red mask)
    # We need to keep it that way (white on black) to match MNIST
    # So we DON'T invert here

    h, w = roi.shape
    scale = size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)
    delta_w = size - new_w
    delta_h = size - new_h
    top, bottom = delta_h // 2, delta_h - delta_h // 2
    left, right = delta_w // 2, delta_w - delta_w // 2
    # Pad with black (0) to match MNIST's black background
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0
    )
    normalized = padded.astype("float32") / 255.0
    return normalized.reshape(1, size, size, 1)


def preprocess_for_pytorch(digit_img):
    """
    Convert preprocessed digit to PyTorch tensor.

    Args:
        digit_img: Output from preprocess_char (shape: 1, 28, 28, 1)

    Returns:
        PyTorch tensor ready for model prediction
    """
    # Remove extra dimensions and get 28x28 array
    img_array = digit_img.squeeze()  # Shape: (28, 28)

    # Apply MNIST normalization
    normalized = (img_array - 0.1307) / 0.3081

    # Convert to PyTorch tensor with correct shape (1, 1, 28, 28)
    tensor = torch.from_numpy(normalized).float().unsqueeze(0).unsqueeze(0)

    return tensor, img_array


# 9. Test with custom red pencil image
print("\n" + "=" * 50)
print("TESTING WITH CUSTOM RED PENCIL IMAGE")
print("=" * 50)

# Replace this with your actual image path
test_image_path = "cropped_grade.png"

try:
    # Extract red digits from image
    print(f"\nExtracting red digits from: {test_image_path}")
    digits = get_grade(test_image_path, display=False)

    if len(digits) == 0:
        print("No red digits detected in the image!")
    else:
        print(f"Found {len(digits)} digit(s)")

        # Show original extracted digits (before preprocessing)
        print("\n--- BEFORE PREPROCESSING ---")
        fig1, axes1 = plt.subplots(1, len(digits), figsize=(3 * len(digits), 3))
        if len(digits) == 1:
            axes1 = [axes1]

        for i, digit_img in enumerate(digits):
            axes1[i].imshow(digit_img, cmap="gray")
            axes1[i].set_title(f"Original Digit {i + 1}\n(from red mask)")
            axes1[i].axis("off")

        plt.suptitle(
            "Extracted Red Digits (Before Preprocessing)",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.show()

        # Show preprocessed digits (after preprocessing)
        print("\n--- AFTER PREPROCESSING ---")
        fig2, axes2 = plt.subplots(1, len(digits), figsize=(3 * len(digits), 3))
        if len(digits) == 1:
            axes2 = [axes2]

        preprocessed_images = []
        for i, digit_img in enumerate(digits):
            # Preprocess digit
            preprocessed = preprocess_char(digit_img)
            preprocessed_images.append(preprocessed)

            # Display preprocessed (inverted and centered)
            display_img = preprocessed.squeeze()  # Remove extra dimensions
            axes2[i].imshow(display_img, cmap="gray")
            axes2[i].set_title(f"Preprocessed Digit {i + 1}\n(28x28, centered)")
            axes2[i].axis("off")

        plt.suptitle(
            "Preprocessed Digits (After Preprocessing - Ready for Model)",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.show()

        # Process and predict each digit
        print("\n--- PREDICTIONS ---")
        predictions = []
        fig3, axes3 = plt.subplots(1, len(digits), figsize=(3 * len(digits), 3))
        if len(digits) == 1:
            axes3 = [axes3]

        model.eval()
        for i, preprocessed in enumerate(preprocessed_images):
            tensor, display_img = preprocess_for_pytorch(preprocessed)
            tensor = tensor.to(device)

            # Make prediction
            with torch.no_grad():
                output = model(tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                predicted_digit = output.argmax(dim=1).item()
                confidence = probabilities[0, predicted_digit].item() * 100

            predictions.append(predicted_digit)

            # Display with prediction
            axes3[i].imshow(display_img, cmap="gray")
            axes3[i].set_title(
                f"Digit {i + 1}\nPrediction: {predicted_digit}\nConfidence: {confidence:.1f}%",
                fontweight="bold",
                color="green" if confidence > 90 else "orange",
            )
            axes3[i].axis("off")

            print(f"\nDigit {i + 1}:")
            print(f"  Predicted: {predicted_digit}")
            print(f"  Confidence: {confidence:.2f}%")
            print(f"  Top 3 predictions:")
            top3_probs, top3_indices = torch.topk(probabilities[0], 3)
            for j, (prob, idx) in enumerate(zip(top3_probs, top3_indices)):
                print(f"    {j + 1}. Digit {idx.item()}: {prob.item() * 100:.2f}%")

        plt.suptitle(
            "Final Predictions with Confidence", fontsize=14, fontweight="bold"
        )
        plt.tight_layout()
        plt.show()

        # Show complete number if multiple digits
        if len(predictions) > 1:
            complete_number = "".join(map(str, predictions))
            print(f"\n{'=' * 50}")
            print(f"COMPLETE NUMBER: {complete_number}")
            print(f"{'=' * 50}")
        elif len(predictions) == 1:
            print(f"\n{'=' * 50}")
            print(f"DETECTED DIGIT: {predictions[0]}")
            print(f"{'=' * 50}")

except FileNotFoundError:
    print(f"\nNote: Could not find test image at '{test_image_path}'")
    print("To test with your own image:")
    print("1. Take a photo of red pencil-written number(s) on white paper")
    print("2. Save it as 'red_pencil_number.jpg' in the same directory")
    print("3. Run this script again")
    print("\nAlternatively, modify 'test_image_path' variable to point to your image.")
except Exception as e:
    print(f"\nError processing image: {e}")
    import traceback

    traceback.print_exc()

# 10. Optional: Test with MNIST test set to verify model works
print("\n" + "=" * 50)
print("TESTING WITH MNIST TEST SET (Verification)")
print("=" * 50)

# Show some predictions from MNIST test set
n_samples = 5
test_iter = iter(test_loader)
images, labels = next(test_iter)

fig, axes = plt.subplots(1, n_samples, figsize=(15, 3))
model.eval()
with torch.no_grad():
    for i in range(n_samples):
        img = images[i : i + 1].to(device)
        output = model(img)
        pred_digit = output.argmax(dim=1).item()
        true_digit = labels[i].item()

        # Denormalize for display
        img_display = images[i].squeeze().numpy()
        img_display = img_display * 0.3081 + 0.1307

        axes[i].imshow(img_display, cmap="gray")
        axes[i].set_title(f"True: {true_digit}\nPred: {pred_digit}")
        axes[i].axis("off")

plt.tight_layout()
plt.show()

print("\nScript completed successfully!")
