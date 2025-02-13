import pandas as pd
import numpy as np
import os

class DataPreprocessor:
    def __init__(self):
        self.data = None
        
    def preprocess(self, input_path, output_path):
        """Preprocess data and save"""
        print("\n[INFO] Starting data preprocessing...")
        
        try:
            # 1. Load data
            self.data = pd.read_csv(input_path)
            print(f"Loaded raw data: {len(self.data)} records")
            
            # 2. Handle missing values first
            self.data = self.data.fillna({
                'discounted_price': 0,
                'actual_price': 0,
                'rating_count': 0,
                'category': 'Unknown',
                'review_content': ''
            })
            
            # 3. Convert price columns
            # First convert to string and clean
            self.data['discounted_price'] = self.data['discounted_price'].astype(str)
            self.data['actual_price'] = self.data['actual_price'].astype(str)
            
            # Remove currency symbols and commas
            self.data['discounted_price'] = self.data['discounted_price'].replace('[\₹,]', '', regex=True)
            self.data['actual_price'] = self.data['actual_price'].replace('[\₹,]', '', regex=True)
            
            # Convert to float
            self.data['discounted_price'] = pd.to_numeric(self.data['discounted_price'], errors='coerce')
            self.data['actual_price'] = pd.to_numeric(self.data['actual_price'], errors='coerce')
            
            # 4. Convert rating_count
            self.data['rating_count'] = self.data['rating_count'].astype(str)
            self.data['rating_count'] = self.data['rating_count'].replace('[,]', '', regex=True)
            self.data['rating_count'] = pd.to_numeric(self.data['rating_count'], errors='coerce')
            
            # 5. Rename columns
            column_mapping = {
                'category': 'main_category',
                'review_content': 'reviews'
            }
            self.data = self.data.rename(columns=column_mapping)
            
            # 6. Filter invalid values
            print("\nRecords before filtering:", len(self.data))
            
            # Remove rows with invalid values
            valid_data = (
                (self.data['discounted_price'] > 0) &
                (self.data['actual_price'] > 0) &
                (self.data['rating_count'] > 0)
            )
            self.data = self.data[valid_data]
            
            print("Records after filtering:", len(self.data))
            
            # 7. Save processed data
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            self.data.to_csv(output_path, index=False)
            print(f"\nProcessed data saved: {len(self.data)} records")
            
            return self.data
            
        except Exception as e:
            print(f"\n[ERROR] Data preprocessing failed: {str(e)}")
            if self.data is not None:
                print("\nFirst few rows of raw data:")
                print(self.data.head())
                print("\nData types:")
                print(self.data.dtypes)
            raise 