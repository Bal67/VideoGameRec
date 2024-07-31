import os
import pandas as pd
import numpy as np


#Loads dataset into Pandas dataframe
def load_dataset(dataset_path):
    # Load the dataset
    df = pd.read_csv(dataset_path)
    
    # Renumber the rows
    df.reset_index(drop=True, inplace=True)
    
    return df


#Extracts relevant features from the dataset and returns a new dataframe
def extract_features(df):
    # Rename columns for clarity
    df.columns = ['user_id', 'game_title', 'action', 'hours', 'flag']
    
    # Convert game titles to lowercase
    df['game_title'] = df['game_title'].str.lower()
    
    # Filter to include only records where the game has been played for 2 or more hours and the action is 'play'
    df = df[(df['hours'] >= 2) & (df['action'] == 'play')]
    
    # Filter games that have been played by at least 20 unique players
    df = df[df.groupby('game_title')['user_id'].transform('count') >= 20]
    
    # Convert game_id to string (assuming game_id is present in the original dataframe)
    df['user_id'] = df['user_id'].astype(str)
    
    # Calculate average hours played per game
    average = df.groupby(['game_title'], as_index=False)['hours'].mean()
    average['avg_hourplayed'] = average['hours']
    average.drop(columns='hours', inplace=True)
    
    # Merge the average hours back to the main dataframe
    df = df.merge(average, on='game_title')
    
    # Define conditions for converting hours to ratings
    condition = [
        df['hours'] >= (0.8 * df['avg_hourplayed']),
        (df['hours'] >= 0.6 * df['avg_hourplayed']) & (df['hours'] < 0.8 * df['avg_hourplayed']),
        (df['hours'] >= 0.4 * df['avg_hourplayed']) & (df['hours'] < 0.6 * df['avg_hourplayed']),
        (df['hours'] >= 0.2 * df['avg_hourplayed']) & (df['hours'] < 0.4 * df['avg_hourplayed']),
        df['hours'] < 0.2 * df['avg_hourplayed']
    ]
    values = [5, 4, 3, 2, 1]
    
    # Apply conditions to create ratings
    df['rating'] = np.select(condition, values)
    
     # Add user features
    user_features = df.groupby('user_id').agg(
        total_games_played=('game_title', 'count'),
        total_hours_played=('hours', 'sum'),
        avg_user_rating=('rating', 'mean')
    ).reset_index()
    
    df = df.merge(user_features, on='user_id', how='left')
    
    # Add game features
    game_features = df.groupby('game_title').agg(
        num_unique_players=('user_id', 'nunique'),
        total_hours_played_on_game=('hours', 'sum'),
        game_popularity=('user_id', 'count')
    ).reset_index()
    
    df = df.merge(game_features, on='game_title', how='left')
    
    # Drop the unnecessary columns
    df = df.drop(columns=['flag', 'avg_hourplayed', 'action'])

    return df




# Save processed dataset
if __name__ == "__main__":
    dataset_path = './datasets/steam/steam-200k.csv'
    
    # Load and renumber the dataset
    df = load_dataset(dataset_path)
    
    # Extract features
    df = extract_features(df)
    
     # Save processed dataset
    processed_data_path = './data/processed_data.csv'
    df.to_csv(processed_data_path, index=False)
    
    # Display the first few rows of the modified dataset
    print(f"Processed dataset saved to {processed_data_path}")

    # Display the first few rows of the modified dataset
    print("Dataset after feature extraction:")
    print(df.head())
