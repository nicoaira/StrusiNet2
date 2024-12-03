import torch.nn as nn
from torch_geometric.nn import GINConv, Set2Set

class GINModel(nn.Module):
    def __init__(self, graph_encoding, hidden_dim, output_dim, dropout=0.1):
        super(GINModel, self).__init__()

        input_dim = 1 if graph_encoding == "allocator" else 7

        # Define GIN convolution layers
        net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.conv = GINConv(net)

        # Define pooling layer
        self.pooling = Set2Set(hidden_dim, processing_steps=10)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Final projection layer for embedding
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # *2 due to Set2Set doubling feature dim

    def forward_once(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Pass through GIN convolution layer
        x = self.conv(x, edge_index)
        x = self.dropout(x)

        # Pooling for graph-level embedding
        x = self.pooling(x, batch)

        # Final projection to embedding space
        embedding = self.fc(x)

        return embedding

    def forward(self, anchor, positive, negative):
        # Forward pass for each of the triplet inputs
        anchor_out = self.forward_once(anchor)
        positive_out = self.forward_once(positive)
        negative_out = self.forward_once(negative)

        return anchor_out, positive_out, negative_out