import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights

class SiameseResNetLSTM(nn.Module):
    def __init__(self, input_channels, hidden_dim, lstm_layers=1):
        super(SiameseResNetLSTM, self).__init__()

        # Load a pre-trained ResNet-50 model
        self.resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        # Modify the first layer to accept contact matrix input (1 channel instead of 3 for RGB)
        self.resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Remove the final fully connected layer from ResNet (we don't need it)
        self.resnet.fc = nn.Identity()

        # LSTM for sequential modeling after CNN, input size should match the ResNet output channels
        self.lstm = nn.LSTM(input_size=2048, hidden_size=hidden_dim, num_layers=lstm_layers, batch_first=True)

        # A fully connected layer for embedding space projection
        self.fc = nn.Linear(hidden_dim, 256)

    def forward_once(self, x):
        # Forward pass through ResNet
        x = self.resnet(x)  # Output shape: (batch_size, 2048)

        # Add a dummy sequence length dimension to pass into the LSTM
        x = x.unsqueeze(1)  # Shape: (batch_size, 1, 2048)

        # Pass through LSTM layer
        x, _ = self.lstm(x)  # LSTM expects (batch_size, sequence_length, input_size)

        # Use the final LSTM output for embedding projection
        x = self.fc(x[:, -1, :])  # Take the last LSTM output for embedding

        return x

    def forward(self, anchor, positive, negative):
        # Forward pass for each of the triplet inputs
        anchor_out = self.forward_once(anchor)
        positive_out = self.forward_once(positive)
        negative_out = self.forward_once(negative)

        return anchor_out, positive_out, negative_out