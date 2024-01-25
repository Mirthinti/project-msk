# File: neural_network.py
"""
Neural Network Training Script

This script defines a simple neural network using PyTorch,
trains it on synthetic data, and saves the trained model.

Author: Your Name
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize a simple neural network.

        Args:
            input_size (int): Number of input features.
            hidden_size (int): Number of neurons in the hidden layer.
            output_size (int): Number of output classes.
        """
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Forward pass through the neural network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def train_neural_network(model, input_data, labels, num_epochs=100, learning_rate=0.01):
    """
    Train the neural network.

    Args:
        model (nn.Module): PyTorch model to be trained.
        input_data (torch.Tensor): Input data.
        labels (torch.Tensor): Target labels.
        num_epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(input_data)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Save the trained model
    torch.save(model.state_dict(), 'simple_nn_model.pth')

if __name__ == "__main__":
    # Dummy data and labels
    input_size = 10
    hidden_size = 20
    output_size = 5
    batch_size = 64

    input_data = torch.randn(batch_size, input_size)
    labels = torch.randint(0, output_size, (batch_size,))

    # Initialize and train the model
    model = SimpleNN(input_size, hidden_size, output_size)
    train_neural_network(model, input_data, labels)
