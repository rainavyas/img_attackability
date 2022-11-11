import torch.nn as nn

class SingleLinear(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.layer = nn.Linear(input_size, num_classes)
    def forward(self, X):
        return self.layer(X)

class FCN(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Linear(input_size, num_classes)
        )
    def forward(self, X):
        return self.model(X)