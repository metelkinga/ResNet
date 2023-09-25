import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms, datasets


transform_resnet = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

train_data_path = "data/train"
test_data_path = "data/test"

batch_size_resnet = 32
train_dataset_resnet = datasets.ImageFolder(train_data_path, transform=transform_resnet)
train_loader_resnet = DataLoader(train_dataset_resnet, batch_size=batch_size_resnet, shuffle=True)

test_dataset_resnet = datasets.ImageFolder(test_data_path, transform=transform_resnet)
test_loader_resnet = DataLoader(test_dataset_resnet, batch_size=batch_size_resnet, shuffle=False)

model_resnet = torchvision.models.resnet18(pretrained=True)

model_resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

num_features_resnet = model_resnet.fc.in_features
model_resnet.fc = nn.Linear(num_features_resnet, 2)

criterion_resnet = nn.CrossEntropyLoss()
optimizer_resnet = optim.Adam(model_resnet.parameters(), lr=0.001)

correct_resnet = 0
total_resnet = 0

num_epochs_resnet = 10
for epoch in range(num_epochs_resnet):
    model_resnet.train()
    running_loss_resnet = 0.0
    for inputs_resnet, labels_resnet in train_loader_resnet:
        optimizer_resnet.zero_grad()
        outputs_resnet = model_resnet(inputs_resnet)
        loss_resnet = criterion_resnet(outputs_resnet, labels_resnet)
        loss_resnet.backward()
        optimizer_resnet.step()
        running_loss_resnet += loss_resnet.item()

        _, predicted_resnet = torch.max(outputs_resnet, 1)
        total_resnet += labels_resnet.size(0)
        correct_resnet += (predicted_resnet == labels_resnet).sum().item()

    accuracy_resnet = correct_resnet / total_resnet
    print(f"Epoch {epoch + 1}/{num_epochs_resnet}, Loss: {running_loss_resnet}, Training Accuracy: {100 * accuracy_resnet:.2f}%")

model_resnet.eval()
correct_resnet = 0
total_resnet = 0
with torch.no_grad():
    for inputs_resnet, labels_resnet in test_loader_resnet:
        outputs_resnet = model_resnet(inputs_resnet)
        _, predicted_resnet = torch.max(outputs_resnet, 1)
        total_resnet += labels_resnet.size(0)
        correct_resnet += (predicted_resnet == labels_resnet).sum().item()
accuracy_resnet = correct_resnet / total_resnet
print(f"Test Accuracy for ResNet: {100 * accuracy_resnet:.2f}%")
