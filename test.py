import torch
import torchvision
import torchvision.models as models
import torch.nn as nn
from torch import optim
import torchvision.transforms as transforms

# Pretrained ResNet-18
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
# Modify first conv layer to accept 1 channel instead of 3
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
# Copy pretrained weights averaged over RGB channels
with torch.no_grad():
    model.conv1.weight = nn.Parameter(model.conv1.weight.mean(dim=1, keepdim=True))
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # expects 224×224
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

trainset = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

for epoch in range(3):
    image_no = 0
    for images, labels in trainloader:
        print(f"epoch number {epoch}", f"image number {image_no}/{len(trainloader)}")
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        image_no += 1

    print(f"Epoch [{epoch + 1}/3], Loss: {loss.item():.4f}")

# save the weigths
torch.save(model.state_dict(), "mnist_resnet18.pth")
