import torch
from torch.utils.data import Dataset
from src.utils import center_pad_matrix, dot_bracket_to_contact_matrix

class TripletRNADataset(Dataset):
    def __init__(self, dataframe, max_len):
        self.dataframe = dataframe
        self.max_len = max_len

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Extract the triplet (anchor, positive, negative)
        anchor_structure = self.dataframe.iloc[idx]["structure_A"]
        positive_structure = self.dataframe.iloc[idx]["structure_P"]
        negative_structure = self.dataframe.iloc[idx]["structure_N"]

        # Pad the structures to max_len and convert to contact matrices
        anchor_padded = center_pad_matrix(anchor_structure, self.max_len)
        positive_padded = center_pad_matrix(positive_structure, self.max_len)
        negative_padded = center_pad_matrix(negative_structure, self.max_len)

        # Convert the dot-bracket structures to contact matrices
        anchor_matrix = dot_bracket_to_contact_matrix(anchor_padded)
        positive_matrix = dot_bracket_to_contact_matrix(positive_padded)
        negative_matrix = dot_bracket_to_contact_matrix(negative_padded)

        # Convert the matrices to PyTorch tensors and cast to float32
        anchor_tensor = torch.tensor(anchor_matrix, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        positive_tensor = torch.tensor(positive_matrix, dtype=torch.float32).unsqueeze(0)
        negative_tensor = torch.tensor(negative_matrix, dtype=torch.float32).unsqueeze(0)

        return anchor_tensor, positive_tensor, negative_tensor