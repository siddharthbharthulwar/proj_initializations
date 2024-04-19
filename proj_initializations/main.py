# import pickle
# import matplotlib.pyplot as plt
# import numpy as np

# with open('data/cifar-10-batches-py/data_batch_1', 'rb') as f:
#     dict = pickle.load(f, encoding='bytes')

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

# Define the neural network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3072, 1028)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(1028, 128)
        self.fc4 = nn.Linear(128, 10)

        # MuP initialization for the last layer with custom alpha
        alpha = 1
        depth = 2  # Number of layers in the network
        power = alpha
        scale = (1 / depth) ** (1 / power)
        nn.init.xavier_uniform_(self.fc4.weight)
        self.fc4.weight.data.mul_(scale)
        nn.init.constant_(self.fc4.bias, 0.0)


    def forward(self, x):
        x = x.view(-1, 3072)  # Flatten the input image
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        return x

# Load the CIFAR-10 dataset
train_dataset = CIFAR10(root='./data', train=True, transform=ToTensor(), download=True)
test_dataset = CIFAR10(root='./data', train=False, transform=ToTensor())

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Create an instance of the neural network
net = Net()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# Training loop
num_epochs = 30
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

