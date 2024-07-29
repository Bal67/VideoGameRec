import os
import pandas as pd
import requests

def download_dataset(url, download_path='./datasets/steam', dataset_file='steam-200k.csv'):
    """
    Downloads the dataset from the specified URL.
    
    Args:
        url (str): URL of the dataset.
        download_path (str): Path to download the dataset to.
        dataset_file (str): Filename of the dataset to save as.
    """
    # Ensure the download path exists
    if not os.path.exists(download_path):
        os.makedirs(download_path)
    
    dataset_path = os.path.join(download_path, dataset_file)
    
    # Download the dataset
    response = requests.get(url)
    with open(dataset_path, 'wb') as file:
        file.write(response.content)
    print(f"Dataset downloaded from {url} to {dataset_path}")

def load_dataset(dataset_path):
    """
    Loads the dataset into a pandas DataFrame.
    
    Args:
        dataset_path (str): Path to the dataset file (e.g., './datasets/steam/steam-200k.csv').
        
    Returns:
        pd.DataFrame: Loaded dataset.
    """
    df = pd.read_csv(dataset_path)
    return df

if __name__ == "__main__":
    url = 'https://github.com/Bal67/VideoGameRec/blob/main/data/steam-200k.csv?raw=true'
    dataset_file = 'steam-200k.csv'
    download_path = './datasets/steam'
    dataset_path = os.path.join(download_path, dataset_file)
    
    download_dataset(url, download_path, dataset_file)
    df = load_dataset(dataset_path)
    
    # Display the first few rows of the dataset
    print(df.head())
