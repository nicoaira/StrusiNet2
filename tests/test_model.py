import torch
import unittest
from src.model.siamese_model import SiameseResNetLSTM
from src.utils import pad_and_convert_to_contact_matrix



class TestSiameseModel(unittest.TestCase):
    def setUp(self):
        # Set up model and dummy data for testing
        self.input_channels = 1
        self.hidden_dim = 256
        self.lstm_layers = 1
        self.max_len = 641
        self.device = 'cpu'

        # Instantiate the model
        self.model = SiameseResNetLSTM(input_channels=self.input_channels, hidden_dim=self.hidden_dim, lstm_layers=self.lstm_layers)
        self.model.to(self.device)
        self.model.eval()

        # Example dot-bracket structure
        self.dot_bracket_structure = "((((...))))...((((....))))"

    def test_pad_and_convert_to_contact_matrix(self):
        # Test if the contact matrix is generated correctly
        contact_matrix = pad_and_convert_to_contact_matrix(self.dot_bracket_structure, self.max_len)
        self.assertEqual(contact_matrix.shape, (self.max_len, self.max_len))

    def test_model_forward_once(self):
        # Test if the model generates an embedding without errors
        contact_matrix = pad_and_convert_to_contact_matrix(self.dot_bracket_structure, self.max_len)
        contact_tensor = torch.tensor(contact_matrix, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)  # Shape: (1, 1, max_len, max_len)
        
        with torch.no_grad():
            embedding = self.model.forward_once(contact_tensor)
            
        # Ensure the output is of expected shape (256,)
        self.assertEqual(embedding.shape, torch.Size([1, self.hidden_dim]))


if __name__ == '__main__':
    unittest.main()