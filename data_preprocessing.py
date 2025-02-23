import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Function to load raw data from CSV (replace with your actual data)
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Function to preprocess the data
def preprocess_data(data):
    # Example preprocessing steps (you should customize this according to your data)
    data = data.dropna()  # Drop missing values
    data = data.reset_index(drop=True)  # Reset index after dropping rows

    # Assuming we have a target column 'penalty_occurred' and we need to encode it
    data['penalty_occurred'] = data['penalty_occurred'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    # Example of feature engineering: you can add features here
    # data['feature_x'] = ... (add your own feature engineering here)

    return data

# Function to split data into features (X) and target (y)
def split_data(data):
    X = data.drop('penalty_occurred', axis=1)  # Features
    y = data['penalty_occurred']  # Target variable
    return X, y
