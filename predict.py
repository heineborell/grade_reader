from mnistmodel import MNISTNet
from mnistmodel import train_epoch, evaluate
from manip import get_grade, preprocess_char, preprocess_for_pytorch
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

# 3. Initialize model and load weights
model = MNISTNet().to(device)
model.load_state_dict(torch.load("mnist_model.pth", map_location=device))
model.eval()  # Set to evaluation mode

print("Model loaded successfully!")

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
        # print("\n--- BEFORE PREPROCESSING ---")
        # fig1, axes1 = plt.subplots(1, len(digits), figsize=(3 * len(digits), 3))
        # if len(digits) == 1:
        #     axes1 = [axes1]
        #
        # for i, digit_img in enumerate(digits):
        #     axes1[i].imshow(digit_img, cmap="gray")
        #     axes1[i].set_title(f"Original Digit {i + 1}\n(from red mask)")
        #     axes1[i].axis("off")
        #
        # plt.suptitle(
        #     "Extracted Red Digits (Before Preprocessing)",
        #     fontsize=14,
        #     fontweight="bold",
        # )
        # plt.tight_layout()
        # plt.show()

        # Show preprocessed digits (after preprocessing)
        # print("\n--- AFTER PREPROCESSING ---")
        # fig2, axes2 = plt.subplots(1, len(digits), figsize=(3 * len(digits), 3))
        # if len(digits) == 1:
        #     axes2 = [axes2]

        preprocessed_images = []
        for i, digit_img in enumerate(digits):
            # Preprocess digit
            preprocessed = preprocess_char(digit_img)
            preprocessed_images.append(preprocessed)

            # Display preprocessed (inverted and centered)
            # display_img = preprocessed.squeeze()  # Remove extra dimensions
            # axes2[i].imshow(display_img, cmap="gray")
            # axes2[i].set_title(f"Preprocessed Digit {i + 1}\n(28x28, centered)")
            # axes2[i].axis("off")

        # plt.suptitle(
        #     "Preprocessed Digits (After Preprocessing - Ready for Model)",
        #     fontsize=14,
        #     fontweight="bold",
        # )
        # plt.tight_layout()
        # plt.show()

        # Process and predict each digit
        # print("\n--- PREDICTIONS ---")
        predictions = []
        # fig3, axes3 = plt.subplots(1, len(digits), figsize=(3 * len(digits), 3))
        # if len(digits) == 1:
        #     axes3 = [axes3]

        model.eval()
        for i, preprocessed in enumerate(preprocessed_images):
            tensor, display_img = preprocess_for_pytorch(preprocessed)
            tensor = tensor.to(device)

            print("came to model")
            # Make prediction
            with torch.no_grad():
                output = model(tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                predicted_digit = output.argmax(dim=1).item()
                confidence = probabilities[0, predicted_digit].item() * 100

            predictions.append(predicted_digit)

            # Display with prediction
            # axes3[i].imshow(display_img, cmap="gray")
            # axes3[i].set_title(
            #     f"Digit {i + 1}\nPrediction: {predicted_digit}\nConfidence: {confidence:.1f}%",
            #     fontweight="bold",
            #     color="green" if confidence > 90 else "orange",
            # )
            # axes3[i].axis("off")

            print(f"\nDigit {i + 1}:")
            print(f"  Predicted: {predicted_digit}")
            print(f"  Confidence: {confidence:.2f}%")
            print(f"  Top 3 predictions:")
            top3_probs, top3_indices = torch.topk(probabilities[0], 3)
            for j, (prob, idx) in enumerate(zip(top3_probs, top3_indices)):
                print(f"    {j + 1}. Digit {idx.item()}: {prob.item() * 100:.2f}%")

        # plt.suptitle(
        #     "Final Predictions with Confidence", fontsize=14, fontweight="bold"
        # )
        # plt.tight_layout()
        # plt.show()

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

# # 10. Optional: Test with MNIST test set to verify model works
# print("\n" + "=" * 50)
# print("TESTING WITH MNIST TEST SET (Verification)")
# print("=" * 50)

# # Show some predictions from MNIST test set
# n_samples = 5
# test_iter = iter(test_loader)
# images, labels = next(test_iter)
#
# fig, axes = plt.subplots(1, n_samples, figsize=(15, 3))
# model.eval()
# with torch.no_grad():
#     for i in range(n_samples):
#         img = images[i : i + 1].to(device)
#         output = model(img)
#         pred_digit = output.argmax(dim=1).item()
#         true_digit = labels[i].item()
#
#         # Denormalize for display
#         img_display = images[i].squeeze().numpy()
#         img_display = img_display * 0.3081 + 0.1307
#
#         axes[i].imshow(img_display, cmap="gray")
#         axes[i].set_title(f"True: {true_digit}\nPred: {pred_digit}")
#         axes[i].axis("off")
#
# plt.tight_layout()
# plt.show()
#
# print("\nScript completed successfully!")
