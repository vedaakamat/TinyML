import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Multi-Layer Perceptron for Activity Classification
    """

    def __init__(self):
        super().__init__()

        # Hidden Layer 1
        self.fc1 = nn.Linear(6, 64)

        # Hidden Layer 2
        self.fc2 = nn.Linear(64, 32)

        # Output Layer
        self.fc3 = nn.Linear(32, 5)

        # Activation Function
        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.relu(self.fc1(x))

        x = self.relu(self.fc2(x))

        x = self.fc3(x)

        return x


# -------------------------------------------------------
# Test Model
# -------------------------------------------------------

if __name__ == "__main__":

    model = MLP()

    print(model)