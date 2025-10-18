from mnistmodel import MNISTNet
from mnistmodel import train_epoch, evaluate
from manip import get_grade, preprocess_char, preprocess_for_pytorch
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


def predict_digit(test_image_path):
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

    try:
        # Extract red digits from image
        print(f"\nExtracting red digits from: {test_image_path}")
        digits = get_grade(test_image_path, display=False)

        if len(digits) == 0:
            print("No red digits detected in the image!")
        else:
            print(f"Found {len(digits)} digit(s)")

            preprocessed_images = []
            for i, digit_img in enumerate(digits):
                # Preprocess digit
                preprocessed = preprocess_char(digit_img)
                preprocessed_images.append(preprocessed)

            predictions = []

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

                print(f"\nDigit {i + 1}:")
                print(f"  Predicted: {predicted_digit}")
                print(f"  Confidence: {confidence:.2f}%")
                print(f"  Top 3 predictions:")
                top3_probs, top3_indices = torch.topk(probabilities[0], 3)
                for j, (prob, idx) in enumerate(zip(top3_probs, top3_indices)):
                    print(f"    {j + 1}. Digit {idx.item()}: {prob.item() * 100:.2f}%")

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
        print(
            "\nAlternatively, modify 'test_image_path' variable to point to your image."
        )
    except Exception as e:
        print(f"\nError processing image: {e}")
        import traceback

        traceback.print_exc()
    return int(complete_number)
