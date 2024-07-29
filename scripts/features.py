import os
import pandas as pd

def load_dataset(dataset_path):
    """
    Loads the dataset into a pandas DataFrame and renumbers the rows.
    
    Args:
        dataset_path (str): Path to the dataset file (e.g., './datasets/steam/steam-200k.csv').
        
    Returns:
        pd.DataFrame: Loaded and renumbered dataset.
    """
    # Load the dataset
    df = pd.read_csv(dataset_path)
    
    # Renumber the rows
    df.reset_index(drop=True, inplace=True)
    
    return df

def extract_features(df):
    """
    Extracts relevant features from the dataset.
    
    Args:
        df (pd.DataFrame): The loaded dataset.
        
    Returns:
        pd.DataFrame: The dataset with extracted features.
    """
    # Rename columns for clarity
    df.columns = ['user_id', 'game_title', 'action', 'hours', 'flag']
    
    # Convert game titles to lowercase
    df['game_title'] = df['game_title'].str.lower()
    
    # Display the first few rows of the dataset to verify
    print("First few rows of the dataset after renaming columns and converting titles to lowercase:")
    print(df.head())
    
    return df

if __name__ == "__main__":
    dataset_path = './datasets/steam/steam-200k.csv'
    
    # Load and renumber the dataset
    df = load_dataset(dataset_path)
    
    # Extract features
    df = extract_features(df)
    
    # Display the first few rows of the modified dataset
    print("Dataset after feature extraction:")
    print(df.head())
