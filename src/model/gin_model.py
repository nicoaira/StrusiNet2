import torch.nn as nn
from torch_geometric.nn import global_add_pool
from torch_geometric.nn import GINConv, Set2Set

class GINModel(nn.Module):
    def __init__(self, hidden_dim, output_dim, graph_encoding = "standard", gin_layers=1, dropout=0.1):
        super(GINModel, self).__init__()

        input_dim = 1 if graph_encoding == "standard" else 7

        # Define GIN MLP
        convs = []
        for i in range(gin_layers):
            if i == 0:
                net = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )
            else:
                net = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )
            convs.append(GINConv(net))
        
        self.convs = nn.ModuleList(convs)
       
        # Define pooling layer option
        self.pooling = global_add_pool
        ## self.pooling = Set2Set(hidden_dim, processing_steps=10)

        # Dropout layer (optional to avoid overfitting)
        ## self.dropout = nn.Dropout(dropout)

        # Fully connected layer to project to embedding space
        self.fc = nn.Linear(hidden_dim, output_dim)
        #if pooling is Set2Set
        ## self.fc = nn.Linear(hidden_dim * 2, output_dim)  # *2 due to Set2Set doubling feature dim

    def forward_once(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Pass through GIN convolution layer
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            #x = self.dropout(x)

        # Pooling for graph-level embedding
        x = self.pooling(x, batch)
        #x = self.dropout(x)

        # Final projection to embedding space
        embedding = self.fc(x)

        return embedding

    def forward(self, anchor, positive, negative):
        # Forward pass for each of the triplet inputs
        anchor_out = self.forward_once(anchor)
        positive_out = self.forward_once(positive)
        negative_out = self.forward_once(negative)

        return anchor_out, positive_out, negative_out