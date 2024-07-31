import os
import pandas as pd
import requests

def download_dataset(url, download_path='./datasets/steam', dataset_file='steam-200k.csv'):
    # Ensure the download path exists
    if not os.path.exists(download_path):
        os.makedirs(download_path)
    
    dataset_path = os.path.join(download_path, dataset_file)
    
    # Download the dataset
    response = requests.get(url)
    with open(dataset_path, 'wb') as file:
        file.write(response.content)
    print(f"Dataset downloaded from {url} to {dataset_path}")

#Loads dataset into Pandas dataframe
def load_dataset(dataset_path):
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
