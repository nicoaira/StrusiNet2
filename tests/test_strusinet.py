import os
import subprocess
import pandas as pd
import unittest

class TestStrusiNet(unittest.TestCase):
    def setUp(self):
        # Setup paths
        self.input_csv = 'example_data/sample_dataset.csv'
        self.output_csv = 'example_data/sample_dataset_with_embeddings.csv'
        self.model_path = 'saved_model/ResNet-Secondary.pth'
    
    def test_generate_embeddings(self):
        # Run the strusinet script using subprocess
        command = [
            'python', 'strusinet.py',
            '--input_csv', self.input_csv,
            '--output_csv', self.output_csv,
            '--structure_column_num', '6',  # Since we know secondary structure is in column 6
            '--header', 'True', 
            '--model_path', self.model_path,
            '--device', 'cpu'
        ]

        result = subprocess.run(command, capture_output=True, text=True)
        
        # Ensure the script ran without errors
        self.assertEqual(result.returncode, 0, f"Script failed with error: {result.stderr}")

        # Check that the output CSV exists
        self.assertTrue(os.path.exists(self.output_csv), "Output CSV file was not created.")
        
        # Load the output CSV and check for 'embedding_vector' column
        df = pd.read_csv(self.output_csv)
        self.assertIn('embedding_vector', df.columns, "Output CSV does not contain 'embedding_vector' column.")

    def tearDown(self):
        # Clean up generated output file
        if os.path.exists(self.output_csv):
            os.remove(self.output_csv)

if __name__ == '__main__':
    unittest.main()
