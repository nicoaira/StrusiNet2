from torch.utils.data import Dataset
from src.utils import dotbracket_to_graph, graph_to_tensor

class GINRNADataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        dot_bracket = self.dataframe.iloc[idx]['secondary_structure']
        G = dotbracket_to_graph(dot_bracket)
        data = graph_to_tensor(G)
        return data
