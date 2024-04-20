import torch
import torch.nn as nn
import torch.optim as optim
from torchviz import make_dot
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

class Net(nn.Module):

    def __init__(self, gamma, net_width, num_classes=10):
        super(Net, self).__init__()
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv1 = nn.Conv2d(3, net_width // 2, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(net_width // 2, net_width, kernel_size=5, stride=2, padding=1)

        self.fc = nn.Linear(net_width, num_classes)
        #MuP initialization
        last_layer_var = gamma * (6 / (net_width)) ** 0.5
        nn.init.normal_(self.fc.weight, mean=0.0, std=torch.sqrt(torch.tensor(last_layer_var)))

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Load the CIFAR-10 dataset
train_dataset = CIFAR10(root='./data', train=True, transform=ToTensor(), download=True)
test_dataset = CIFAR10(root='./data', train=False, transform=ToTensor())

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Create an instance of the neural network
net = Net(0.1, 196)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    for images, labels in train_loader:
        # Forward pass
        outputs = net(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Evaluation on the test set
net.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy on the test set: {accuracy:.2f}%")

