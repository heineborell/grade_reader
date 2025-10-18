import mnistmodel
from mnistmodel import train_epoch, evaluate
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

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
model = mnistmodel.MNISTNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("\nModel architecture:")
print(model)


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
