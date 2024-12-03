import torch
import torch.nn as nn
from torch_geometric.nn import GINConv
from torch_geometric.data import Data

class GINModel2Layers(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(GINModel2Layers, self).__init__()

        # Define GIN layers
        self.conv1 = GINConv(nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        ))

        self.conv2 = GINConv(nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        ))

        # Fully connected layer to project to embedding space
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward_once(self, data):
        x, edge_index = data.x, data.edge_index

        # Pass data through GIN layers
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)

        # Sum pooling for graph-level embedding
        x = torch.sum(x, dim=0)  # sum pooling

        # Final projection to embedding space
        embedding = self.fc(x)
        return embedding
    
    def forward(self, anchor, positive, negative):
        # Forward pass for each of the triplet inputs
        anchor_out = self.forward_once(anchor)
        positive_out = self.forward_once(positive)
        negative_out = self.forward_once(negative)

        return anchor_out, positive_out, negative_out