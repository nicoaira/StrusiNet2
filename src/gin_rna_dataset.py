from torch.utils.data import Dataset
from src.utils import dotbracket_to_graph, graph_to_tensor

class GINRNADataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        anchor_structure = self.dataframe.iloc[idx]["structure_A"]
        positive_structure = self.dataframe.iloc[idx]["structure_P"]
        negative_structure = self.dataframe.iloc[idx]["structure_N"]

        g_anchor = dotbracket_to_graph(anchor_structure)
        g_positive = dotbracket_to_graph(positive_structure)
        g_negative = dotbracket_to_graph(negative_structure)

        data_anchor = graph_to_tensor(g_anchor)
        data_positive = graph_to_tensor(g_positive)
        data_negative = graph_to_tensor(g_negative)
        
        return data_anchor, data_positive, data_negative
