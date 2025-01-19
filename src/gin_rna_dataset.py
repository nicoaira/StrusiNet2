from torch.utils.data import Dataset
from src.utils import dotbracket_to_forgi_graph, dotbracket_to_graph, forgi_graph_to_tensor, graph_to_tensor

class GINRNADataset(Dataset):
    def __init__(self, dataframe, graph_encoding = "standard"):
        self.dataframe = dataframe
        self.graph_encoding = graph_encoding

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        anchor_structure = self.dataframe.iloc[idx]["structure_A"]
        positive_structure = self.dataframe.iloc[idx]["structure_P"]
        negative_structure = self.dataframe.iloc[idx]["structure_N"]

        if self.graph_encoding == "standard":
            g_anchor = dotbracket_to_graph(anchor_structure)
            g_positive = dotbracket_to_graph(positive_structure)
            g_negative = dotbracket_to_graph(negative_structure)

            data_anchor = graph_to_tensor(g_anchor)
            data_positive = graph_to_tensor(g_positive)
            data_negative = graph_to_tensor(g_negative)
        
        elif self.graph_encoding == "forgi":
            g_anchor = dotbracket_to_forgi_graph(anchor_structure)
            g_positive = dotbracket_to_forgi_graph(positive_structure)
            g_negative = dotbracket_to_forgi_graph(negative_structure)

            data_anchor = forgi_graph_to_tensor(g_anchor)
            data_positive = forgi_graph_to_tensor(g_positive)
            data_negative = forgi_graph_to_tensor(g_negative)
        
        return data_anchor, data_positive, data_negative
