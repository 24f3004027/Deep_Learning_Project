import pandas as pd
import numpy as np

try:
    df = pd.read_csv("data/train.csv")
    print(f"Dataset Dimensions: {df.shape}")
    print("\nMissing values per column:")
    print(df.isnull().sum())
    print("\nAnswer distribution:")
    print(df['answer'].value_counts(normalize=True))
except Exception as e:
    print(f"Error loading dataset: {e}")
