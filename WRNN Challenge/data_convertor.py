import os
import pandas as pd
from typing import List
from sklearn.model_selection import train_test_split

class DataConvertor:
    def __init__(self, input_file: str, output_dir: str):
        self.input_file = input_file
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def load_data(self) -> pd.DataFrame:
        """Load data from a parquet file."""
        return pd.read_parquet(self.input_file)

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the data (e.g., handle missing values)."""
        df = df.dropna()  # Example: drop rows with missing values
        return df

    def split_data(self, df: pd.DataFrame, test_size: float = 0.2):
        """Split the data into training and testing sets."""
        return train_test_split(df, test_size=test_size, random_state=42)

    def save_to_csv(self, df: pd.DataFrame, filename: str):
        """Save DataFrame to a CSV file."""
        filepath = os.path.join(self.output_dir, filename)
        df.to_csv(filepath, index=False)

    def convert(self):
        """Main method to convert parquet to CSV with preprocessing and splitting."""
        df = self.load_data()
        df = self.preprocess_data(df)
        train_df, test_df = self.split_data(df)
        self.save_to_csv(train_df, 'train.csv')
        self.save_to_csv(test_df, 'test.csv')

if __name__ == "__main__":
    from pathlib import Path

    input_file = Path("E:/wunder_challenge/competition_package/datasets/train.parquet")
    output_dir = Path("E:/wunder_challenge/competition_package")

    convertor = DataConvertor(input_file, output_dir)
    convertor.convert()
    print(f"✅ Data converted and saved to {output_dir}")
